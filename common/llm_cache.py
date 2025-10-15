import os
import json
import pickle
import hashlib
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np


class EnhancedLLMCacheManager:
    """
    Enhanced LLM cache management system with dynamic expert support,
    configuration hashing, and compatibility features.
    """

    def __init__(self, cache_dir: str = "llm_cache", max_cache_size: int = 10000, **kwargs):
        """
        Initialize the LLM cache manager.

        Args:
            cache_dir: Directory to store cache files
            max_cache_size: Maximum number of cached responses
        """
        # Handle both parameter names for backward compatibility
        if 'llm_cache_dir' in kwargs:
            cache_dir = kwargs['llm_cache_dir']

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size

        # In-memory cache for faster access
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_access_times: Dict[str, float] = {}

        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.saves = 0

    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Generate a unique hash for the configuration to ensure cache compatibility.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration hash string
        """
        # Sort the config to ensure consistent hashing
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, dataset: str, config_hash: str) -> Path:
        """Get the cache file path for a dataset and configuration."""
        return self.cache_dir / f"{dataset}_{config_hash}.pkl"

    def _find_compatible_cache(self, dataset: str, expert_num: int) -> Optional[Dict[str, Any]]:
        """
        Find compatible cache files when exact match is not found.
        Uses heuristic based on expert number similarity.

        Args:
            dataset: Dataset name
            expert_num: Number of experts in current configuration

        Returns:
            Compatible cache data if found, None otherwise
        """
        # Look for cache files with the same dataset prefix
        pattern = f"{dataset}_*.pkl"
        cache_files = list(self.cache_dir.glob(pattern))

        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                # Check compatibility based on expert number
                cached_expert_num = cache_data.get('config', {}).get('expert_num', 0)

                # Compatible if expert numbers are the same or cached has more experts
                if cached_expert_num >= expert_num:
                    print(f"Found compatible cache: {cache_file.name} "
                          f"(cached: {cached_expert_num}, current: {expert_num})")
                    return cache_data
            except Exception as e:
                print(f"Error reading cache file {cache_file}: {e}")
                continue

        return None

    def load_expert_preferences(self, dataset: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load expert preferences from cache with dynamic expert support.

        Args:
            dataset: Dataset name
            config: Model configuration

        Returns:
            Cached expert preferences dictionary
        """
        config_hash = self._generate_config_hash(config)
        cache_path = self._get_cache_path(dataset, config_hash)

        # Check in-memory cache first
        memory_key = f"{dataset}_{config_hash}"
        if memory_key in self.memory_cache:
            self.hits += 1
            self.cache_access_times[memory_key] = time.time()
            return self.memory_cache[memory_key]

        # Try to load from disk
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                self.hits += 1
                self._store_in_memory(memory_key, cache_data)
                return cache_data
            except Exception as e:
                print(f"Error loading cache from {cache_path}: {e}")

        # Try to find compatible cache
        compatible_cache = self._find_compatible_cache(dataset, config.get('expert_num', 0))
        if compatible_cache:
            self.hits += 1
            # Adjust for different expert numbers if needed
            current_expert_num = config.get('expert_num', 0)
            cached_expert_num = compatible_cache.get('config', {}).get('expert_num', 0)

            if current_expert_num < cached_expert_num:
                # Truncate expert preferences to match current configuration
                preferences = compatible_cache.get('preferences', {})
                adjusted_preferences = {}
                for node_id, pref_data in preferences.items():
                    if 'expert' in pref_data:
                        expert = pref_data['expert']
                        if expert < current_expert_num:
                            adjusted_preferences[node_id] = pref_data
                        else:
                            # Map to a valid expert if possible
                            adjusted_preferences[node_id] = {
                                **pref_data,
                                'expert': expert % current_expert_num
                            }
                    else:
                        adjusted_preferences[node_id] = pref_data

                compatible_cache['preferences'] = adjusted_preferences
                compatible_cache['config']['expert_num'] = current_expert_num

            self._store_in_memory(memory_key, compatible_cache)
            return compatible_cache

        self.misses += 1
        return {'preferences': {}, 'config': config, 'metadata': {}}

    def _store_in_memory(self, key: str, data: Dict[str, Any]):
        """Store data in memory cache with LRU management."""
        # Check if cache is full
        if len(self.memory_cache) >= self.max_cache_size:
            self._evict_lru()

        self.memory_cache[key] = data
        self.cache_access_times[key] = time.time()

    def _evict_lru(self):
        """Evict least recently used item from memory cache."""
        if not self.cache_access_times:
            return

        lru_key = min(self.cache_access_times.items(), key=lambda x: x[1])[0]
        del self.memory_cache[lru_key]
        del self.cache_access_times[lru_key]

    def save_expert_preferences(self, dataset: str, config: Dict[str, Any],
                               preferences: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """
        Save expert preferences to cache.

        Args:
            dataset: Dataset name
            config: Model configuration
            preferences: Expert preferences dictionary
            metadata: Additional metadata to store
        """
        config_hash = self._generate_config_hash(config)
        cache_path = self._get_cache_path(dataset, config_hash)

        cache_data = {
            'preferences': preferences,
            'config': config,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'version': '1.0'
        }

        # Store in memory
        memory_key = f"{dataset}_{config_hash}"
        self._store_in_memory(memory_key, cache_data)

        # Save to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            self.saves += 1
        except Exception as e:
            print(f"Error saving cache to {cache_path}: {e}")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'saves': self.saves,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_files': len(list(self.cache_dir.glob("*.pkl")))
        }

    def clear_cache(self, dataset: Optional[str] = None):
        """
        Clear cache for a specific dataset or all caches.

        Args:
            dataset: Dataset name to clear, if None clears all
        """
        if dataset is None:
            # Clear all cache
            self.memory_cache.clear()
            self.cache_access_times.clear()
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print("Cleared all cache files")
        else:
            # Clear specific dataset cache
            pattern = f"{dataset}_*.pkl"
            cache_files = list(self.cache_dir.glob(pattern))

            # Clear from memory
            keys_to_remove = [key for key in self.memory_cache.keys() if key.startswith(f"{dataset}_")]
            for key in keys_to_remove:
                del self.memory_cache[key]
                if key in self.cache_access_times:
                    del self.cache_access_times[key]

            # Clear from disk
            for cache_file in cache_files:
                cache_file.unlink()

            print(f"Cleared cache for dataset: {dataset} ({len(cache_files)} files)")

    def optimize_cache(self):
        """Optimize cache by removing old or invalid entries."""
        current_time = time.time()
        max_age = 7 * 24 * 3600  # 7 days

        # Remove old entries from memory
        keys_to_remove = []
        for key, access_time in self.cache_access_times.items():
            if current_time - access_time > max_age:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.memory_cache[key]
            del self.cache_access_times[key]

        # Remove old files from disk
        for cache_file in self.cache_dir.glob("*.pkl"):
            if current_time - cache_file.stat().st_mtime > max_age:
                cache_file.unlink()
                print(f"Removed old cache file: {cache_file}")

    def list_cached_datasets(self) -> List[str]:
        """List all datasets with cached preferences."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        datasets = set()

        for cache_file in cache_files:
            # Extract dataset name from filename
            parts = cache_file.stem.split('_')
            if len(parts) >= 2:
                dataset = '_'.join(parts[:-1])  # Everything except the hash
                datasets.add(dataset)

        return sorted(list(datasets))


class LLMResponseParser:
    """
    Robust LLM response parser with multiple fallback strategies.
    """

    @staticmethod
    def parse_llm_response_robust(response: str, expert_num: int) -> Dict[str, Any]:
        """
        Robustly parse LLM response with multiple fallback strategies.

        Args:
            response: LLM response string
            expert_num: Number of experts to validate against

        Returns:
            Parsed response dictionary with expert ranking and confidence
        """
        if not response or not isinstance(response, str):
            return {"expert": 0, "confidence": 0.5, "reason": "Invalid response"}

        response = response.strip()

        # Strategy 1: Direct JSON parsing
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                if "expert" in parsed and isinstance(parsed["expert"], int):
                    expert = parsed["expert"]
                    confidence = parsed.get("confidence", 0.5)
                    if 0 <= expert < expert_num:
                        return {
                            "expert": expert,
                            "confidence": float(confidence),
                            "reason": parsed.get("reason", ""),
                            "strategy": "direct_json"
                        }
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract expert number with regex
        try:
            # Look for patterns like "expert": 2 or expert:2 or Expert 2
            expert_patterns = [
                r'"expert":\s*(\d+)',
                r'expert:\s*(\d+)',
                r'Expert\s+(\d+)',
                r'selected expert[^0-9]*(\d+)',
                r'chosen expert[^0-9]*(\d+)'
            ]

            for pattern in expert_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    expert = int(match.group(1))
                    if 0 <= expert < expert_num:
                        # Try to extract confidence
                        confidence_match = re.search(r'confidence[^\d]*(\d+\.?\d*)', response, re.IGNORECASE)
                        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                        confidence = min(1.0, max(0.0, confidence))  # Clamp to [0,1]

                        return {
                            "expert": expert,
                            "confidence": confidence,
                            "reason": f"Extracted using regex pattern: {pattern}",
                            "strategy": "regex_extraction"
                        }
        except Exception:
            pass

        # Strategy 3: Look for expert mentions in text
        try:
            expert_mentions = []
            for i in range(expert_num):
                patterns = [f"expert {i}", f"expert-{i}", f"e{i}", f"expert{i}"]
                for pattern in patterns:
                    if pattern.lower() in response.lower():
                        expert_mentions.append(i)
                        break

            if expert_mentions:
                # Use the most mentioned expert
                from collections import Counter
                expert_counts = Counter(expert_mentions)
                expert = expert_counts.most_common(1)[0][0]

                return {
                    "expert": expert,
                    "confidence": 0.6,  # Moderate confidence for text-based extraction
                    "reason": f"Expert {expert} mentioned most frequently in text",
                    "strategy": "text_mention"
                }
        except Exception:
            pass

        # Strategy 4: Random fallback
        import random
        random_expert = random.randint(0, expert_num - 1)

        return {
            "expert": random_expert,
            "confidence": 0.1,  # Very low confidence for random
            "reason": "Random fallback - no valid expert found in response",
            "strategy": "random_fallback"
        }

    @staticmethod
    def parse_dpo_preferences(response: str, expert_num: int) -> List[Tuple[int, int]]:
        """
        Parse DPO (Direct Preference Optimization) preferences from LLM response.

        Args:
            response: LLM response containing expert ranking
            expert_num: Number of experts

        Returns:
            List of preference pairs (preferred_expert, less_preferred_expert)
        """
        # Try to extract ranking list
        ranking_patterns = [
            r'"ranking":\s*\[([^\]]+)\]',
            r'ranking:\s*\[([^\]]+)\]',
            r'ranking[^:]*:\s*\[([^\]]+)\]'
        ]

        for pattern in ranking_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    ranking_str = match.group(1)
                    # Extract numbers from the ranking string
                    ranking = [int(x.strip()) for x in re.findall(r'\d+', ranking_str)]

                    # Validate ranking
                    if all(0 <= x < expert_num for x in ranking):
                        # Create preference pairs from ranking
                        preferences = []
                        for i in range(len(ranking) - 1):
                            for j in range(i + 1, len(ranking)):
                                preferences.append((ranking[i], ranking[j]))
                        return preferences
                except Exception:
                    continue

        # Fallback: create random preferences
        preferences = []
        for i in range(expert_num):
            for j in range(i + 1, expert_num):
                if np.random.random() > 0.5:
                    preferences.append((i, j))
                else:
                    preferences.append((j, i))

        return preferences