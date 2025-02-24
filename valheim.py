# ... (Your original ThunderstoreClient class definition) ...
import re
import json
from typing import Dict, Set, List, Optional, Tuple
from difflib import SequenceMatcher  # For Levenshtein distance
import argparse
import os
import time
import requests
import concurrent.futures
from functools import partial
import itertools
from collections import Counter

class ThunderstoreClient:
    def __init__(self, cache_dir="cache"):
        self.api_base_url = "https://thunderstore.io/c/valheim/api/v1/package/"
        self.packages_data = []
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "thunderstore_cache.json")
        self.cache_expiry_time = 24 * 3600  # 24 hours in seconds
        self.all_mod_names = set()  # 新增：存储所有 Mod 名称的集合
        self.timeout = 10  # 10 seconds timeout

    def load_packages_from_cache(self) -> bool:
        """从缓存加载包数据"""
        try:
            if not os.path.exists(self.cache_file):
                return False
            
            # 检查缓存是否过期
            if time.time() - os.path.getmtime(self.cache_file) > self.cache_expiry_time:
                return False
            
            with open(self.cache_file, 'r') as f:
                self.packages_data = json.load(f)
            return bool(self.packages_data)
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False

    def save_packages_to_cache(self) -> bool:
        """保存包数据到缓存"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.packages_data, f)
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False

    def download_packages(self) -> bool:
        """从 Thunderstore API 下载包数据"""
        try:
            print("Downloading mod data from Thunderstore...")
            response = requests.get(self.api_base_url, timeout=self.timeout)
            response.raise_for_status()
            self.packages_data = response.json()
            print("Download completed successfully.")
            return self.save_packages_to_cache()
        except requests.exceptions.Timeout:
            print("Error: Connection to Thunderstore timed out. Please check your internet connection.")
            return False
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Thunderstore. Please check your internet connection.")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error downloading packages: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def get_all_mods(self):
        """获取所有 Valheim Mod 信息，优先从缓存加载，缓存过期则重新下载"""
        print("Loading mod data...")
        if self.load_packages_from_cache():
            print("Loaded mod data from cache.")
            return True
        print("Cache not available or expired, downloading from Thunderstore...")
        return self.download_packages()

    def get_thunderstore_ids(self) -> List[str]:
        """Gets a list of all Thunderstore IDs (full_name)."""
        print("Getting Thunderstore IDs...")
        if not self.packages_data:
            if not self.get_all_mods():  # Load data if needed
                return []  # Return empty list if data loading fails
        ids = [package['full_name'] for package in self.packages_data]
        print(f"Found {len(ids)} Thunderstore mods")
        return ids

    def get_thunderstore_data(self) -> Dict[str, Dict]:
        """获取 Thunderstore 数据，包括版本信息"""
        print("Getting Thunderstore data...")
        if not self.packages_data:
            if not self.get_all_mods():
                return {}
        
        # 创建包含版本信息的数据结构
        ts_data = {}
        for package in self.packages_data:
            full_name = package['full_name']
            versions = [re.sub(r'[^0-9.]', '', v['version_number']) 
                       for v in package['versions']]
            # 过滤掉空版本号
            versions = [v for v in versions if v]
            
            ts_data[full_name] = {
                'versions': versions,
                'latest_version': versions[0] if versions else '',
                'author': package['owner'],
                'name': package['name']
            }
            
        print(f"Found {len(ts_data)} Thunderstore mods")
        return ts_data

# Move LogParser out as a separate class
class LogParser:
    def __init__(self, log_path="output_log.txt"):
        self.log_path = log_path

    def parse_log(self) -> Dict[str, str]:
        """解析日志文件，返回 mod_name -> version 的映射"""
        print(f"Parsing log file: {self.log_path}")
        loading_pattern = re.compile(r"\[Info\s+:\s+BepInEx\]\s+Loading\s+\[([^]]+?)\s+([\d\w\.-]+)\]")
        mods = {}
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    match = loading_pattern.search(line)
                    if match:
                        mod_name, version = match.groups()
                        mod_name = mod_name.strip()
                        version = version.strip()
                        # 清理版本号，移除非数字和点号的字符
                        version = re.sub(r'[^0-9.]', '', version)
                        mods[mod_name] = version
            print(f"Found {len(mods)} mods in log file")
            return mods
        except FileNotFoundError:
            print(f"Error: Log file not found at {self.log_path}")
            return {}
        except Exception as e:
            print(f"Error parsing log file: {e}")
            return {}

class ModNameMatcher:
    def __init__(self, thunderstore_data: Dict[str, Dict]):
        self.thunderstore_data = thunderstore_data
        self.thunderstore_ids = list(thunderstore_data.keys())
        
        # 尝试从缓存加载命名规则分析结果
        self.naming_patterns = self._load_patterns_from_cache()
        if not self.naming_patterns:
            self.analyze_thunderstore_names()
            self._save_patterns_to_cache()
            
        # 预处理 IDs
        self.processed_ids = self._preprocess_thunderstore_ids()

    def _load_patterns_from_cache(self) -> Optional[Dict]:
        """从缓存加载命名规则分析结果"""
        cache_file = "naming_patterns_cache.json"
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _save_patterns_to_cache(self):
        """保存命名规则分析结果到缓存"""
        cache_file = "naming_patterns_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.naming_patterns, f)
        except Exception:
            pass

    def analyze_thunderstore_names(self):
        """分析 Thunderstore ID 的命名规则"""
        print("\nAnalyzing Thunderstore naming patterns...")
        total_ids = len(self.thunderstore_ids)
        
        # 统计模式
        patterns = {
            'all_lowercase': 0,
            'all_uppercase': 0,
            'mixed_case': 0,
            'pascal_case': 0,
            'camel_case': 0,
            'contains_space': 0,
            'contains_hyphen': 0,
            'contains_underscore': 0,
            'contains_special': 0,
        }
        
        # 分析每个ID
        for ts_id in self.thunderstore_ids:
            if '-' in ts_id:
                author, mod_name = ts_id.split('-', 1)
                
                # 分析大小写模式
                if mod_name.islower():
                    patterns['all_lowercase'] += 1
                elif mod_name.isupper():
                    patterns['all_uppercase'] += 1
                elif mod_name[0].isupper() and not any(c.isupper() for c in mod_name[1:]):
                    patterns['pascal_case'] += 1
                elif mod_name[0].islower() and any(c.isupper() for c in mod_name[1:]):
                    patterns['camel_case'] += 1
                else:
                    patterns['mixed_case'] += 1
                
                # 分析分隔符
                if ' ' in mod_name:
                    patterns['contains_space'] += 1
                if '-' in mod_name:
                    patterns['contains_hyphen'] += 1
                if '_' in mod_name:
                    patterns['contains_underscore'] += 1
                
                # 分析特殊字符
                if re.search(r'[^a-zA-Z0-9\-_ ]', mod_name):
                    patterns['contains_special'] += 1

        # 打印分析结果
        print("\nMod Name Pattern Analysis:")
        print("=" * 60)
        print(f"Total IDs analyzed: {total_ids}")
        for pattern, count in patterns.items():
            percentage = (count / total_ids) * 100
            print(f"- {pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

        # 保存分析结果供匹配使用
        self.naming_patterns = patterns

    def _preprocess_thunderstore_ids(self) -> Dict[str, str]:
        """基于分析结果预处理 Thunderstore IDs"""
        processed = {}
        for ts_id in self.thunderstore_ids:
            if '-' in ts_id:
                author, mod_name = ts_id.split('-', 1)
                # 存储完整ID
                processed[ts_id] = ts_id
                # 存储不带作者的mod名称
                processed[mod_name] = ts_id
                # 存储无分隔符版本（保持原始大小写）
                processed[mod_name.replace('-', '')] = ts_id
                processed[mod_name.replace('_', '')] = ts_id
        return processed

    def find_matches(self, keyword: str, version: str = None) -> Tuple[List[str], Dict[str, float]]:
        """查找所有可能的匹配，考虑版本信息"""
        matches = []
        scores = {}

        # 1. 直接匹配
        if keyword in self.processed_ids:
            match_id = self.processed_ids[keyword]
            matches.append(match_id)
            scores[match_id] = self._calculate_match_score(match_id, keyword, version)

        # 2. 移除分隔符后匹配
        normalized = keyword.replace(' ', '').replace('-', '').replace('_', '')
        for ts_id, original_id in self.processed_ids.items():
            if original_id not in matches:  # 避免重复
                ts_normalized = ts_id.replace(' ', '').replace('-', '').replace('_', '')
                if normalized == ts_normalized:
                    matches.append(original_id)
                    scores[original_id] = self._calculate_match_score(original_id, keyword, version)

        # 3. 相似度匹配
        if len(matches) < 3:  # 如果还没找到足够的匹配
            for ts_id in self.thunderstore_ids:
                if ts_id not in matches:
                    if '-' in ts_id:
                        _, mod_name = ts_id.split('-', 1)
                        similarity = self._similarity_score(normalized, 
                                                         mod_name.replace('-', '').replace('_', ''))
                        if similarity > 0.8:
                            matches.append(ts_id)
                            base_score = similarity
                            scores[ts_id] = self._calculate_match_score(ts_id, keyword, version, base_score)

        # 按分数排序
        matches.sort(key=lambda x: scores[x], reverse=True)
        return matches, scores

    def _calculate_match_score(self, ts_id: str, keyword: str, version: str = None, 
                             base_score: float = 1.0) -> float:
        """计算匹配分数，考虑版本信息"""
        score = base_score
        
        # 如果没有版本信息，直接返回基础分数
        if not version or ts_id not in self.thunderstore_data:
            return score
            
        ts_versions = self.thunderstore_data[ts_id]['versions']
        if not ts_versions:
            return score
            
        # 版本完全匹配
        if version in ts_versions:
            score = min(1.0, score + 0.2)  # 版本完全匹配加0.2分
            return score
            
        # 部分匹配（比较主版本号和次版本号）
        version_parts = version.split('.')
        for ts_version in ts_versions:
            ts_version_parts = ts_version.split('.')
            matching_parts = 0
            for i in range(min(len(version_parts), len(ts_version_parts))):
                if version_parts[i] == ts_version_parts[i]:
                    matching_parts += 1
                else:
                    break
            if matching_parts > 0:
                # 根据匹配的版本号部分数量增加分数
                version_score = 0.1 * (matching_parts / max(len(version_parts), len(ts_version_parts)))
                score = min(1.0, score + version_score)
                break
                
        return score

    def generate_mapping(self, keywords_with_versions: Dict[str, str]) -> Dict[str, List[tuple[str, float]]]:
        """生成所有关键字的映射，包含版本信息"""
        mapping = {}
        
        for keyword, version in keywords_with_versions.items():
            matches, scores = self.find_matches(keyword, version)
            mapping[keyword] = [(match, scores[match]) for match in matches]
            
        return mapping

    def _similarity_score(self, s1: str, s2: str) -> float:
        """计算两个字符串的相似度"""
        # 1. 完全匹配
        if s1 == s2:
            return 1.0
            
        # 2. 长度差异过大时快速返回
        len_diff = abs(len(s1) - len(s2))
        if len_diff > min(len(s1), len(s2)) / 2:
            return 0.0
            
        # 3. 使用序列匹配器计算基础相似度
        similarity = SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
        
        # 4. 检查单词匹配情况
        words1 = s1.lower().replace('_', ' ').replace('-', ' ').split()
        words2 = s2.lower().replace('_', ' ').replace('-', ' ').split()
        
        # 计算单词匹配的精确度
        if words1 and words2:
            # 如果目标字符串包含额外的修饰词（如reborn, extended等），降低其分数
            if len(words2) > len(words1):
                extra_words = set(words2) - set(words1)
                if extra_words:
                    similarity = max(0.0, similarity - 0.1 * len(extra_words))
            
            # 如果所有核心单词都匹配（不考虑顺序），增加分数
            if all(w in words2 for w in words1):
                similarity = min(1.0, similarity + 0.15)
                
                # 如果目标字符串没有额外单词，进一步增加分数
                if len(words1) == len(words2):
                    similarity = min(1.0, similarity + 0.1)
            
        # 5. 调整分数
        # 如果字符串较长（>10字符）且相似度较高（>0.8），给予额外加分
        if len(s1) > 10 and len(s2) > 10 and similarity > 0.8:
            similarity = min(1.0, similarity + 0.1)
            
        return similarity

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Match Valheim mod names from log file to Thunderstore IDs')
    parser.add_argument('--log-file', '-l', 
                       default="output_log.txt",
                       help='Path to the BepInEx log file (default: output_log.txt)')
    
    args = parser.parse_args()

    # 1. 获取 Thunderstore 数据（包含版本信息）
    thunderstore_client = ThunderstoreClient()
    thunderstore_data = thunderstore_client.get_thunderstore_data()

    if not thunderstore_data:
        print("Could not retrieve Thunderstore data. Exiting.")
        exit()

    # 2. 创建 matcher 实例
    matcher = ModNameMatcher(thunderstore_data)

    # 3. 解析日志文件（现在包含版本信息）
    log_parser = LogParser(log_path=args.log_file)
    mods_with_versions = log_parser.parse_log()

    if not mods_with_versions:
        print("No mods found in log file. Exiting.")
        exit()

    # 4. 生成映射
    mapping = matcher.generate_mapping(mods_with_versions)

    # 5. 打印结果
    print("\nMatching Results:")
    print("=" * 80)
    
    # 按匹配状态分类
    matched_mods = []  # 确定的匹配（包括单一匹配和唯一版本匹配）
    unmatched_mods = []
    multiple_matches = []
    
    # 先处理所有匹配
    for keyword, matches in mapping.items():
        if not matches:
            unmatched_mods.append(keyword)
            continue
            
        if len(matches) == 1:
            matched_mods.append((keyword, matches[0]))
            continue
            
        # 检查多重匹配中的版本匹配情况
        log_version = mods_with_versions[keyword]
        version_matches = []
        
        for match, score in matches:
            ts_versions = thunderstore_data[match]['versions']
            if log_version in ts_versions:
                version_matches.append((match, score))
        
        # 如果只有一个版本匹配，将其视为确定的匹配
        if len(version_matches) == 1:
            matched_mods.append((keyword, version_matches[0]))
        else:
            multiple_matches.append((keyword, matches))
    
    # 打印统计信息
    total = len(mapping)
    print(f"Total mods: {total}")
    print(f"Matched: {len(matched_mods)} | Multiple: {len(multiple_matches)} | Unmatched: {len(unmatched_mods)}")
    print("=" * 80)
    
    # 只打印需要处理的多重匹配结果
    if multiple_matches:
        print("\nMultiple Matches (Need Review):")
        print("-" * 80)
        for keyword, matches in multiple_matches:
            log_version = mods_with_versions[keyword]
            print(f"? {keyword} (version: {log_version})")
            print(f"  Found {len(matches)} possible matches:")
            
            sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
            for i, (match, score) in enumerate(sorted_matches, 1):
                author, mod_name = match.split('-', 1)
                prefix = "  ├─" if i < len(sorted_matches) else "  └─"
                ts_versions = thunderstore_data[match]['versions']
                version_info = ""
                if log_version in ts_versions:
                    version_info = " [Version Match]"
                else:
                    version_info = f" [Available versions: {', '.join(ts_versions[:3])}...]"
                print(f"{prefix} {author}-{mod_name} (score: {score:.2f}){version_info}")


