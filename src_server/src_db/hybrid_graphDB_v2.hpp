#ifndef HYBRID_GRAPH_DB_V2_H
#define HYBRID_GRAPH_DB_V2_H

#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <algorithm> 
#include <utility> 
#include <list>
#include <unordered_map>
#include <tuple>
#include <random>
#include <regex>

// RocksDB 헤더
#include <rocksdb/db.h>
#include <rocksdb/options.h>

struct HybridNode {
    int id;
    std::string text_content;
    std::vector<float> embedding;
    std::vector<int> neighbors;

    void print_info() const;
    std::string serialize() const;
    static HybridNode deserialize(const std::string& data);
};

class HybridGraphDB {
private:
    std::unique_ptr<rocksdb::DB> db;
    std::string db_path;

    float similarity_threshold; 
    size_t cache_capacity; 

    // LRU Cache System
    std::list<HybridNode> lru_list;
    std::unordered_map<int, std::list<HybridNode>::iterator> lru_map;

    void update_cache(const HybridNode& node);
    float calculate_cosine_similarity(const std::vector<float>& vec_a, const std::vector<float>& vec_b);
    void save_node_to_rocksdb(const HybridNode& node);

    // ✅ 새 내부 함수들
    bool is_text_complete(const std::string& text) const;
    std::string clean_and_validate_text(const std::string& text, int min_length = 200) const;

public:
    HybridGraphDB(std::string path = "/tmp/my_hybrid_db", float threshold = 0.7f, size_t capacity = 2000);
    ~HybridGraphDB();

    void add_node(int id, const std::string& text, const std::vector<float>& embedding);
    HybridNode get_node(int id);

    int get_node_count();
    std::vector<int> get_all_node_ids();

    // 엣지 관리
    void add_edge(int src_id, int dst_id, float weight);
    std::vector<std::pair<int, float>> get_neighbors(int src_id);
    std::vector<std::tuple<int, int, float>> get_all_edges();

    void clear_cache();

    // ✅ 새 공개 함수들
    std::string get_db_path() const { return db_path; }
    
    /**
     * @brief 전체 그래프의 유사도 기반 엣지를 생성합니다.
     * @param threshold 유사도 임계값 (기본: 0.7)
     * @param sample_size 각 노드당 비교할 최대 샘플 수 (기본: 1000)
     * @param batch_size 한 번에 처리할 노드 수 (기본: 100)
     */
    void build_similarity_graph(float threshold = 0.7f, int sample_size = 1000, int batch_size = 100);

    /**
     * @brief 두 노드 간 코사인 유사도를 계산합니다.
     */
    float get_similarity(int id1, int id2);

    /**
     * @brief 그래프 통계를 출력합니다.
     */
    void print_graph_stats() const;

    /**
     * @brief 텍스트 품질을 검증하고 문제 있는 노드를 찾습니다.
     * @param min_length 최소 텍스트 길이 (기본: 200자)
     * @return 문제 있는 노드 ID 리스트
     */
    std::vector<int> validate_text_quality(int min_length = 200);

    /**
     * @brief 고아 노드(엣지 없는 노드)를 찾습니다.
     */
    std::vector<int> find_orphan_nodes();

    /**
     * @brief 특정 노드들에 대해서만 엣지를 생성합니다.
     * @param node_ids 대상 노드 ID 리스트
     */
    void build_edges_for_nodes(const std::vector<int>& node_ids, float threshold = 0.7f, int sample_size = 500);

    /**
     * @brief k-최근접 이웃으로 엣지를 생성합니다. (더 빠름)
     * @param k 각 노드당 연결할 이웃 수
     */
    void build_knn_graph(int k = 10);
};

#endif // HYBRID_GRAPH_DB_V2_H
