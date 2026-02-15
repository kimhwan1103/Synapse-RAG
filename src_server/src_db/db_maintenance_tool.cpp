#include "hybrid_graphDB_v2.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

/**
 * @brief DB 품질 검증 및 개선 도구
 * 
 * 기능:
 * 1. 텍스트 품질 검증
 * 2. 고아 노드 찾기
 * 3. 유사도 그래프 구축
 * 4. k-NN 그래프 구축
 * 5. 통계 출력
 */

void print_menu() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Hybrid GraphDB - Maintenance Tool                 ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  1. Show Graph Statistics                                  ║\n";
    std::cout << "║  2. Validate Text Quality                                  ║\n";
    std::cout << "║  3. Find Orphan Nodes (no edges)                           ║\n";
    std::cout << "║  4. Build Similarity Graph (slow, comprehensive)           ║\n";
    std::cout << "║  5. Build k-NN Graph (fast, approximate)                   ║\n";
    std::cout << "║  6. Fix Orphan Nodes                                       ║\n";
    std::cout << "║  7. Export Problematic Nodes to File                       ║\n";
    std::cout << "║  8. Clear Cache                                            ║\n";
    std::cout << "║  0. Exit                                                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "Enter choice: ";
}

int main(int argc, char* argv[]) {
    std::string db_path = "/tmp/my_hybrid_db";
    
    if (argc > 1) {
        db_path = argv[1];
    }
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Hybrid GraphDB - Maintenance Tool                 ║\n";
    std::cout << "║                    Version 2.0                             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nConnecting to database: " << db_path << "\n";
    
    HybridGraphDB db(db_path, 0.7f, 2000);
    
    while (true) {
        print_menu();
        
        int choice;
        std::cin >> choice;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        switch (choice) {
            case 0:
                std::cout << "\nExiting...\n";
                return 0;
                
            case 1: {
                // 통계 출력
                db.print_graph_stats();
                break;
            }
            
            case 2: {
                // 텍스트 품질 검증
                int min_length;
                std::cout << "Enter minimum text length (default 200): ";
                std::cin >> min_length;
                if (min_length <= 0) min_length = 200;
                
                auto problematic = db.validate_text_quality(min_length);
                
                std::cout << "\nSample problematic nodes:\n";
                for (int i = 0; i < std::min(10, (int)problematic.size()); ++i) {
                    HybridNode node = db.get_node(problematic[i]);
                    std::cout << "  [" << node.id << "] " 
                             << node.text_content.substr(0, 50) << "...\n";
                }
                break;
            }
            
            case 3: {
                // 고아 노드 찾기
                auto orphans = db.find_orphan_nodes();
                
                std::cout << "\nSample orphan nodes:\n";
                for (int i = 0; i < std::min(10, (int)orphans.size()); ++i) {
                    HybridNode node = db.get_node(orphans[i]);
                    std::cout << "  [" << node.id << "] " 
                             << node.text_content.substr(0, 50) << "...\n";
                }
                break;
            }
            
            case 4: {
                // 유사도 그래프 구축
                float threshold;
                int sample_size, batch_size;
                
                std::cout << "Enter similarity threshold (0.0-1.0, default 0.7): ";
                std::cin >> threshold;
                if (threshold <= 0) threshold = 0.7f;
                
                std::cout << "Enter sample size per node (default 1000): ";
                std::cin >> sample_size;
                if (sample_size <= 0) sample_size = 1000;
                
                std::cout << "Enter batch size (default 100): ";
                std::cin >> batch_size;
                if (batch_size <= 0) batch_size = 100;
                
                std::cout << "\n⚠️  This may take a long time! Continue? (y/n): ";
                char confirm;
                std::cin >> confirm;
                
                if (confirm == 'y' || confirm == 'Y') {
                    db.build_similarity_graph(threshold, sample_size, batch_size);
                } else {
                    std::cout << "Cancelled.\n";
                }
                break;
            }
            
            case 5: {
                // k-NN 그래프 구축
                int k;
                std::cout << "Enter k (number of neighbors, default 10): ";
                std::cin >> k;
                if (k <= 0) k = 10;
                
                std::cout << "\n⚠️  This may take a long time! Continue? (y/n): ";
                char confirm;
                std::cin >> confirm;
                
                if (confirm == 'y' || confirm == 'Y') {
                    db.build_knn_graph(k);
                } else {
                    std::cout << "Cancelled.\n";
                }
                break;
            }
            
            case 6: {
                // 고아 노드 수정
                std::cout << "Finding orphan nodes...\n";
                auto orphans = db.find_orphan_nodes();
                
                if (orphans.empty()) {
                    std::cout << "No orphan nodes found!\n";
                } else {
                    float threshold;
                    int sample_size;
                    
                    std::cout << "Enter similarity threshold (default 0.6): ";
                    std::cin >> threshold;
                    if (threshold <= 0) threshold = 0.6f;
                    
                    std::cout << "Enter sample size (default 500): ";
                    std::cin >> sample_size;
                    if (sample_size <= 0) sample_size = 500;
                    
                    db.build_edges_for_nodes(orphans, threshold, sample_size);
                }
                break;
            }
            
            case 7: {
                // 문제 노드를 파일로 내보내기
                std::string filename;
                std::cout << "Enter output filename (default: problematic_nodes.txt): ";
                std::cin.ignore();
                std::getline(std::cin, filename);
                if (filename.empty()) filename = "problematic_nodes.txt";
                
                auto problematic = db.validate_text_quality(200);
                
                std::ofstream outfile(filename);
                outfile << "Problematic Nodes Report\n";
                outfile << "========================\n\n";
                
                for (int node_id : problematic) {
                    HybridNode node = db.get_node(node_id);
                    outfile << "Node ID: " << node.id << "\n";
                    outfile << "Text Length: " << node.text_content.length() << "\n";
                    outfile << "Text: " << node.text_content << "\n";
                    outfile << "------------------------\n\n";
                }
                
                outfile.close();
                std::cout << "Exported " << problematic.size() << " nodes to " << filename << "\n";
                break;
            }
            
            case 8: {
                // 캐시 정리
                db.clear_cache();
                break;
            }
            
            default:
                std::cout << "Invalid choice!\n";
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "\n[Time] Operation took " << duration.count() << " seconds.\n";
    }
    
    return 0;
}
