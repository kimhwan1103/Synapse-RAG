#include "hybrid_graphDB_v2.hpp"
#include <sstream>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <set>
#include <queue>

#include <rocksdb/table.h>

// =============================================================
// HybridNode Implementation (기존과 동일)
// =============================================================

void HybridNode::print_info() const {
    std::cout << "[Node " << id << "] " << text_content.substr(0, 20) << "...";
    std::cout << " | Neighbors: " << neighbors.size() << "\n";
}

std::string HybridNode::serialize() const {
    std::stringstream ss;
    ss << id << "|" << text_content << "|";
    ss << embedding.size() << "|";
    for(size_t i=0; i<embedding.size(); ++i) {
        ss << embedding[i];
        if(i < embedding.size()-1) ss << ",";
    }
    ss << "|";
    ss << neighbors.size() << "|";
    for(size_t i=0; i<neighbors.size(); ++i) {
        ss << neighbors[i];
        if(i < neighbors.size()-1) ss << ",";
    }
    return ss.str();
}

HybridNode HybridNode::deserialize(const std::string& data) {
    HybridNode node;
    std::stringstream ss(data);
    std::string segment;
    if(!std::getline(ss, segment, '|')) return node;
    try { node.id = std::stoi(segment); } catch(...) { node.id = -1; }
    std::getline(ss, node.text_content, '|');
    if(std::getline(ss, segment, '|')) {
        int emb_size = 0;
        try { emb_size = std::stoi(segment); } catch(...) {}
        std::getline(ss, segment, '|'); 
        std::stringstream vs(segment);
        std::string val;
        for(int i=0; i<emb_size; ++i) {
            std::getline(vs, val, ',');
            try { node.embedding.push_back(std::stof(val)); } catch(...) {}
        }
    }
    if(std::getline(ss, segment, '|')) {
        int nb_size = 0;
        try { nb_size = std::stoi(segment); } catch(...) {}
        if (nb_size > 0) {
            std::getline(ss, segment, '|');
            std::stringstream ns(segment);
            std::string val;
            for(int i=0; i<nb_size; ++i) {
                std::getline(ns, val, ',');
                try { node.neighbors.push_back(std::stoi(val)); } catch(...) {}
            }
        }
    }
    return node;
}

// =============================================================
// HybridGraphDB Implementation
// =============================================================

HybridGraphDB::HybridGraphDB(std::string path, float threshold, size_t capacity) 
    : db_path(path), similarity_threshold(threshold), cache_capacity(capacity) {
    
    rocksdb::Options options;
    options.create_if_missing = true;

    rocksdb::BlockBasedTableOptions table_options;
    table_options.block_cache = rocksdb::NewLRUCache(128 * 1024 * 1024);
    table_options.cache_index_and_filter_blocks = false;
    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));
    
    rocksdb::DB* raw_db;
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &raw_db);

    if (!status.ok()) {
        std::cerr << "RocksDB Open Error: " << status.ToString() << std::endl;
    } else {
        db.reset(raw_db);
        std::cout << "[System] RocksDB Connected (LRU: " << cache_capacity 
                  << ", Thr: " << similarity_threshold << ")\n";
    }
}

HybridGraphDB::~HybridGraphDB() {
    std::cout << "[System] Closing GraphDB..." << std::endl;
}

// ✅ 새 함수: 텍스트 완성도 체크
bool HybridGraphDB::is_text_complete(const std::string& text) const {
    if (text.empty()) return false;
    
    // 마지막 문자 확인
    char last = text.back();
    
    // 완전한 문장 부호로 끝나는지
    if (last == '.' || last == '!' || last == '?' || 
        last == ')' || last == ']' || last == '}' || last == '"') {
        return true;
    }

    // UTF-8 Multibyte suffixes
    if (text.size() >= 3) {
        std::string suffix = text.substr(text.size() - 3);
        if (suffix == "。" || suffix == "、" || suffix == "」") return true;
    }
    
    // ... 으로 끝나면 잘린 것
    if (text.length() >= 3 && text.substr(text.length()-3) == "...") {
        return false;
    }
    
    return false;
}

// ✅ 새 함수: 텍스트 정제 및 검증
std::string HybridGraphDB::clean_and_validate_text(const std::string& text, int min_length) const {
    std::string cleaned = text;
    
    // 1. 앞뒤 공백 제거
    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r"));
    cleaned.erase(cleaned.find_last_not_of(" \t\n\r") + 1);
    
    // 2. 완전한 문장으로 끝나지 않으면 마지막 문장까지만
    if (!is_text_complete(cleaned)) {
        // 마지막 완전한 문장 부호 찾기
        size_t last_period = cleaned.find_last_of(".!?。、");
        if (last_period != std::string::npos && last_period > min_length / 2) {
            cleaned = cleaned.substr(0, last_period + 1);
        } else {
            // 완전한 문장이 없으면 빈 문자열 반환
            return "";
        }
    }
    
    // 3. 최소 길이 체크
    if (cleaned.length() < min_length) {
        return "";
    }
    
    // 4. [Wiki: ...] 같은 접두어는 유지
    
    return cleaned;
}

float HybridGraphDB::calculate_cosine_similarity(const std::vector<float>& vec_a, const std::vector<float>& vec_b) {
    if (vec_a.size() != vec_b.size()) return 0.0f;
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < vec_a.size(); ++i) {
        dot += vec_a[i] * vec_b[i];
        norm_a += vec_a[i] * vec_a[i];
        norm_b += vec_b[i] * vec_b[i];
    }
    if (norm_a == 0 || norm_b == 0) return 0.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

void HybridGraphDB::update_cache(const HybridNode& node) {
    if (lru_map.find(node.id) != lru_map.end()) {
        lru_list.erase(lru_map[node.id]);
        lru_map.erase(node.id);
    }
    if (lru_list.size() >= cache_capacity) {
        int last_id = lru_list.back().id;
        lru_list.pop_back();
        lru_map.erase(last_id);
    }
    lru_list.push_front(node);
    lru_map[node.id] = lru_list.begin();
}

void HybridGraphDB::save_node_to_rocksdb(const HybridNode& node) {
    if (!db) return;
    std::string key = "N_" + std::to_string(node.id);
    std::string value = node.serialize();
    db->Put(rocksdb::WriteOptions(), key, value);
}

void HybridGraphDB::add_node(int id, const std::string& text, const std::vector<float>& embedding) {
    // ✅ 텍스트 정제
    std::string cleaned_text = clean_and_validate_text(text, 100);
    
    if (cleaned_text.empty()) {
        std::cerr << "[Warning] Node " << id << " has invalid text, skipping..." << std::endl;
        return;
    }
    
    std::vector<int> initial_neighbors;

    // 캐시된 노드들과만 비교 (add_node 시점에는 빠른 삽입 우선)
    // 전체 엣지는 build_similarity_graph()로 별도 구축
    for (const auto& cached_node : lru_list) {
        if (cached_node.id == id) continue;
        float sim = calculate_cosine_similarity(cached_node.embedding, embedding);
        
        if (sim >= similarity_threshold) { 
            initial_neighbors.push_back(cached_node.id);
        }
    }
    
    // 시간적 연결 (직전 노드)
    if (!lru_list.empty()) {
        int prev_id = lru_list.front().id;
        bool already_linked = false;
        for(int nid : initial_neighbors) if(nid == prev_id) already_linked = true;
        
        if (!already_linked && prev_id != id) {
            initial_neighbors.push_back(prev_id);
        }
    }

    HybridNode new_node = {id, cleaned_text, embedding, initial_neighbors};
    save_node_to_rocksdb(new_node);
    update_cache(new_node);
}

HybridNode HybridGraphDB::get_node(int id) {
    // 캐시 확인
    if (lru_map.find(id) != lru_map.end()) {
        HybridNode cached_node = *lru_map[id];
        lru_list.erase(lru_map[id]);
        lru_list.push_front(cached_node);
        lru_map[id] = lru_list.begin();
        return cached_node;
    }

    HybridNode node; node.id = -1;
    if (!db) return node;

    // 신버전 키("N_" + ID) 시도
    std::string key_new = "N_" + std::to_string(id);
    std::string value;
    rocksdb::Status s = db->Get(rocksdb::ReadOptions(), key_new, &value);

    // 실패하면 구버전 키(ID only) 시도
    if (!s.ok()) {
        std::string key_old = std::to_string(id);
        s = db->Get(rocksdb::ReadOptions(), key_old, &value);
    }

    if (s.ok()) {
        node = HybridNode::deserialize(value);
        if (node.id == -1) node.id = id;
        update_cache(node);
    }
    return node;
}

void HybridGraphDB::add_edge(int src_id, int dst_id, float weight) {
    if (!db) return;
    std::string key = "E_" + std::to_string(src_id);
    std::string value;
    rocksdb::Status s = db->Get(rocksdb::ReadOptions(), key, &value);
    std::string new_entry = std::to_string(dst_id) + ":" + std::to_string(weight);
    if (s.ok()) value += "," + new_entry; else value = new_entry;
    db->Put(rocksdb::WriteOptions(), key, value);
}

std::vector<std::pair<int, float>> HybridGraphDB::get_neighbors(int src_id) {
    std::vector<std::pair<int, float>> neighbors;
    if (!db) return neighbors;
    std::string key = "E_" + std::to_string(src_id);
    std::string value;
    if (db->Get(rocksdb::ReadOptions(), key, &value).ok()) {
        std::stringstream ss(value);
        std::string segment;
        while (std::getline(ss, segment, ',')) {
            size_t p = segment.find(':');
            if (p != std::string::npos) {
                try { neighbors.push_back({std::stoi(segment.substr(0, p)), std::stof(segment.substr(p+1))}); } catch(...) {}
            }
        }
    }
    return neighbors;
}

int HybridGraphDB::get_node_count() {
    int count = 0;
    if (!db) return 0;
    
    rocksdb::Iterator* it = db->NewIterator(rocksdb::ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::string key = it->key().ToString();
        if (key.rfind("N_", 0) == 0) {
            count++;
        }
        else if (std::all_of(key.begin(), key.end(), ::isdigit)) {
            count++;
        }
    }
    delete it;
    return count;
}

std::vector<std::tuple<int, int, float>> HybridGraphDB::get_all_edges() {
    std::vector<std::tuple<int, int, float>> edges;
    if (!db) return edges;
    rocksdb::Iterator* it = db->NewIterator(rocksdb::ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::string key = it->key().ToString();
        if (key.rfind("E_", 0) == 0) {
            try {
                int src = std::stoi(key.substr(2));
                std::stringstream ss(it->value().ToString());
                std::string seg;
                while (std::getline(ss, seg, ',')) {
                    size_t p = seg.find(':');
                    if (p != std::string::npos) {
                        edges.emplace_back(src, std::stoi(seg.substr(0, p)), std::stof(seg.substr(p+1)));
                    }
                }
            } catch(...) {}
        }
    }
    delete it;
    return edges;
}

void HybridGraphDB::clear_cache() {
    size_t before_size = lru_list.size();
    std::unordered_map<int, std::list<HybridNode>::iterator>().swap(lru_map);
    std::list<HybridNode>().swap(lru_list);
    std::cout << "[Memory] Cache Cleared! Freed " << before_size << " nodes from RAM." << std::endl;
}

std::vector<int> HybridGraphDB::get_all_node_ids() {
    std::vector<int> ids;
    if (!db) return ids;

    ids.reserve(get_node_count()); 

    rocksdb::Iterator* it = db->NewIterator(rocksdb::ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        std::string key = it->key().ToString();
        
        if (key.rfind("N_", 0) == 0) {
            try {
                ids.push_back(std::stoi(key.substr(2)));
            } catch (...) {}
        }
        else if (std::all_of(key.begin(), key.end(), ::isdigit)) {
            try {
                ids.push_back(std::stoi(key));
            } catch (...) {}
        }
    }
    delete it;
    return ids;
}

// ✅ 새 함수: 전체 유사도 그래프 구축
void HybridGraphDB::build_similarity_graph(float threshold, int sample_size, int batch_size) {
    std::cout << "\n============================================================\n";
    std::cout << "Building Similarity Graph\n";
    std::cout << "============================================================\n";
    std::cout << "  Threshold: " << threshold << "\n";
    std::cout << "  Sample Size: " << sample_size << "\n";
    std::cout << "  Batch Size: " << batch_size << "\n";
    std::cout << "============================================================\n\n";
    
    std::vector<int> all_ids = get_all_node_ids();
    int total = all_ids.size();
    int edges_created = 0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < total; i += batch_size) {
        int batch_end = std::min(i + batch_size, total);
        
        std::cout << "  Progress: " << i << "/" << total 
                 << " (" << (100 * i / total) << "%, edges: " << edges_created << ")\r" << std::flush;
        
        // 배치 처리
        for (int j = i; j < batch_end; ++j) {
            HybridNode node_i = get_node(all_ids[j]);
            if (node_i.id == -1) continue;
            
            // 샘플링: 모든 노드 비교는 너무 느림
            std::vector<int> sample_ids;
            if (total - j - 1 > sample_size) {
                // 랜덤 샘플링
                std::vector<int> remaining(all_ids.begin() + j + 1, all_ids.end());
                std::sample(remaining.begin(), remaining.end(), 
                           std::back_inserter(sample_ids), sample_size, gen);
            } else {
                sample_ids.assign(all_ids.begin() + j + 1, all_ids.end());
            }
            
            for (int other_id : sample_ids) {
                HybridNode node_k = get_node(other_id);
                if (node_k.id == -1) continue;
                
                float sim = calculate_cosine_similarity(node_i.embedding, node_k.embedding);
                
                if (sim >= threshold) {
                    add_edge(all_ids[j], other_id, sim);
                    add_edge(other_id, all_ids[j], sim);
                    edges_created += 2;
                }
            }
        }
        
        // 배치마다 캐시 정리
        if (i % (batch_size * 10) == 0) {
            clear_cache();
        }
    }
    
    std::cout << "\n\n[Success] Similarity graph built! Created " << edges_created << " edges.\n";
    std::cout << "============================================================\n\n";
}

// ✅ 새 함수: 두 노드 간 유사도
float HybridGraphDB::get_similarity(int id1, int id2) {
    HybridNode n1 = get_node(id1);
    HybridNode n2 = get_node(id2);
    if (n1.id == -1 || n2.id == -1) return 0.0f;
    return calculate_cosine_similarity(n1.embedding, n2.embedding);
}

// ✅ 새 함수: 그래프 통계
void HybridGraphDB::print_graph_stats() const {
    std::cout << "\n============================================================\n";
    std::cout << "Graph Statistics\n";
    std::cout << "============================================================\n";
    
    // 노드 개수는 const 함수에서 접근 불가하므로 수정 필요
    std::cout << "  Total Nodes: " << const_cast<HybridGraphDB*>(this)->get_node_count() << "\n";
    std::cout << "  Total Edges: " << const_cast<HybridGraphDB*>(this)->get_all_edges().size() << "\n";
    std::cout << "  Cache Size: " << lru_list.size() << "/" << cache_capacity << "\n";
    std::cout << "============================================================\n\n";
}

// ✅ 새 함수: 텍스트 품질 검증
std::vector<int> HybridGraphDB::validate_text_quality(int min_length) {
    std::cout << "\n============================================================\n";
    std::cout << "Validating Text Quality (min_length=" << min_length << ")\n";
    std::cout << "============================================================\n";
    
    std::vector<int> problematic_nodes;
    std::vector<int> all_ids = get_all_node_ids();
    
    int too_short = 0, incomplete = 0, empty = 0;
    
    for (size_t i = 0; i < all_ids.size(); ++i) {
        if (i % 1000 == 0) {
            std::cout << "  Progress: " << i << "/" << all_ids.size() << "\r" << std::flush;
        }
        
        HybridNode node = get_node(all_ids[i]);
        if (node.id == -1) continue;
        
        bool is_problematic = false;
        
        // 빈 텍스트
        if (node.text_content.empty()) {
            empty++;
            is_problematic = true;
        }
        // 너무 짧음
        else if (node.text_content.length() < min_length) {
            too_short++;
            is_problematic = true;
        }
        // 불완전한 문장
        else if (!is_text_complete(node.text_content)) {
            incomplete++;
            is_problematic = true;
        }
        
        if (is_problematic) {
            problematic_nodes.push_back(node.id);
        }
    }
    
    std::cout << "\n\n[Results]\n";
    std::cout << "  Empty texts: " << empty << "\n";
    std::cout << "  Too short: " << too_short << "\n";
    std::cout << "  Incomplete: " << incomplete << "\n";
    std::cout << "  Total problematic: " << problematic_nodes.size() << "/" << all_ids.size() << "\n";
    std::cout << "============================================================\n\n";
    
    return problematic_nodes;
}

// ✅ 새 함수: 고아 노드 찾기
std::vector<int> HybridGraphDB::find_orphan_nodes() {
    std::cout << "\n============================================================\n";
    std::cout << "Finding Orphan Nodes (nodes without edges)\n";
    std::cout << "============================================================\n";
    
    std::vector<int> orphans;
    std::vector<int> all_ids = get_all_node_ids();
    
    for (size_t i = 0; i < all_ids.size(); ++i) {
        if (i % 1000 == 0) {
            std::cout << "  Progress: " << i << "/" << all_ids.size() << "\r" << std::flush;
        }
        
        auto neighbors = get_neighbors(all_ids[i]);
        if (neighbors.empty()) {
            orphans.push_back(all_ids[i]);
        }
    }
    
    std::cout << "\n\n[Results]\n";
    std::cout << "  Orphan nodes: " << orphans.size() << "/" << all_ids.size() << "\n";
    std::cout << "============================================================\n\n";
    
    return orphans;
}

// ✅ 새 함수: 특정 노드들에 대해 엣지 생성
void HybridGraphDB::build_edges_for_nodes(const std::vector<int>& node_ids, float threshold, int sample_size) {
    std::cout << "\n============================================================\n";
    std::cout << "Building Edges for " << node_ids.size() << " Nodes\n";
    std::cout << "============================================================\n";
    
    std::vector<int> all_ids = get_all_node_ids();
    int edges_created = 0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < node_ids.size(); ++i) {
        if (i % 100 == 0) {
            std::cout << "  Progress: " << i << "/" << node_ids.size() 
                     << " (edges: " << edges_created << ")\r" << std::flush;
        }
        
        HybridNode node = get_node(node_ids[i]);
        if (node.id == -1) continue;
        
        // 샘플링
        std::vector<int> sample_ids;
        if (all_ids.size() > sample_size) {
            std::sample(all_ids.begin(), all_ids.end(), 
                       std::back_inserter(sample_ids), sample_size, gen);
        } else {
            sample_ids = all_ids;
        }
        
        for (int other_id : sample_ids) {
            if (other_id == node_ids[i]) continue;
            
            HybridNode other = get_node(other_id);
            if (other.id == -1) continue;
            
            float sim = calculate_cosine_similarity(node.embedding, other.embedding);
            
            if (sim >= threshold) {
                add_edge(node_ids[i], other_id, sim);
                add_edge(other_id, node_ids[i], sim);
                edges_created += 2;
            }
        }
    }
    
    std::cout << "\n\n[Success] Created " << edges_created << " edges.\n";
    std::cout << "============================================================\n\n";
}

// ✅ 새 함수: k-NN 그래프 (더 빠른 방법)
void HybridGraphDB::build_knn_graph(int k) {
    std::cout << "\n============================================================\n";
    std::cout << "Building k-NN Graph (k=" << k << ")\n";
    std::cout << "============================================================\n";
    
    std::vector<int> all_ids = get_all_node_ids();
    int total = all_ids.size();
    int edges_created = 0;
    
    for (int i = 0; i < total; ++i) {
        if (i % 100 == 0) {
            std::cout << "  Progress: " << i << "/" << total 
                     << " (edges: " << edges_created << ")\r" << std::flush;
        }
        
        HybridNode node_i = get_node(all_ids[i]);
        if (node_i.id == -1) continue;
        
        // 우선순위 큐로 top-k 유지 (최소 힙)
        std::priority_queue<std::pair<float, int>, 
                          std::vector<std::pair<float, int>>,
                          std::greater<std::pair<float, int>>> pq;
        
        for (int j = 0; j < total; ++j) {
            if (i == j) continue;
            
            HybridNode node_j = get_node(all_ids[j]);
            if (node_j.id == -1) continue;
            
            float sim = calculate_cosine_similarity(node_i.embedding, node_j.embedding);
            
            if (pq.size() < k) {
                pq.push({sim, all_ids[j]});
            } else if (sim > pq.top().first) {
                pq.pop();
                pq.push({sim, all_ids[j]});
            }
        }
        
        // top-k 이웃들과 엣지 생성
        while (!pq.empty()) {
            auto [sim, neighbor_id] = pq.top();
            pq.pop();
            add_edge(all_ids[i], neighbor_id, sim);
            edges_created++;
        }
    }
    
    std::cout << "\n\n[Success] k-NN graph built! Created " << edges_created << " edges.\n";
    std::cout << "============================================================\n\n";
}
