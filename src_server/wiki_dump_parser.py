#!/usr/bin/env python3
"""
위키피디아 덤프 파일 직접 파싱
- XML 덤프 (kowiki-latest-pages-articles.xml.bz2) 지원
- 전처리 → 재파싱 → JSONL 출력
"""

import os
import re
import bz2
import json
import logging
from typing import Iterator, Dict, List
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class WikiDumpParser:
    """위키피디아 XML 덤프 파서"""
    
    def __init__(self):
        self.stats = {
            'total_pages': 0,
            'valid_pages': 0,
            'skipped_pages': 0,
            'total_chunks': 0
        }
    
    def parse_xml_dump(self, dump_path: str) -> Iterator[Dict[str, str]]:
        """
        XML 덤프를 파싱하여 페이지별로 yield
        
        Args:
            dump_path: 덤프 파일 경로 (.xml.bz2 또는 .xml)
        
        Yields:
            {"title": "...", "text": "..."}
        """
        logger.info(f"📖 Reading dump: {dump_path}")
        
        # 파일 열기 (bz2 압축 자동 감지)
        if dump_path.endswith('.bz2'):
            logger.info("  Format: Compressed (bz2)")
            file_obj = bz2.open(dump_path, 'rt', encoding='utf-8')
        else:
            logger.info("  Format: Plain XML")
            file_obj = open(dump_path, 'r', encoding='utf-8')
        
        # XML 파싱 (간단한 버전 - 전체 파일을 메모리에 올리지 않음)
        current_page = {}
        in_page = False
        in_title = False
        in_text = False
        
        title_buffer = []
        text_buffer = []
        
        try:
            for line in file_obj:
                line = line.strip()
                
                # 페이지 시작
                if '<page>' in line:
                    in_page = True
                    current_page = {}
                    title_buffer = []
                    text_buffer = []
                
                # 페이지 종료
                elif '</page>' in line and in_page:
                    if title_buffer and text_buffer:
                        current_page['title'] = ''.join(title_buffer).strip()
                        current_page['text'] = ''.join(text_buffer).strip()
                        
                        self.stats['total_pages'] += 1
                        
                        # 유효성 검사
                        if self._is_valid_page(current_page):
                            self.stats['valid_pages'] += 1
                            yield current_page
                        else:
                            self.stats['skipped_pages'] += 1
                    
                    in_page = False
                
                # 제목
                elif '<title>' in line:
                    in_title = True
                    # <title>텍스트</title> 형태
                    match = re.search(r'<title>(.*?)</title>', line)
                    if match:
                        title_buffer.append(match.group(1))
                        in_title = False
                    else:
                        title_buffer.append(line.replace('<title>', ''))
                
                elif '</title>' in line and in_title:
                    title_buffer.append(line.replace('</title>', ''))
                    in_title = False
                
                elif in_title:
                    title_buffer.append(line)
                
                # 텍스트
                elif '<text' in line:
                    in_text = True
                    # <text xml:space="preserve">내용</text> 형태
                    match = re.search(r'<text[^>]*>(.*?)(?:</text>)?$', line)
                    if match:
                        content = match.group(1)
                        if '</text>' in line:
                            text_buffer.append(content.replace('</text>', ''))
                            in_text = False
                        else:
                            text_buffer.append(content)
                
                elif '</text>' in line and in_text:
                    text_buffer.append(line.replace('</text>', ''))
                    in_text = False
                
                elif in_text:
                    text_buffer.append(line)
        
        finally:
            file_obj.close()
        
        logger.info(f"✅ Parsed {self.stats['valid_pages']} valid pages out of {self.stats['total_pages']}")
    
    def _is_valid_page(self, page: Dict[str, str]) -> bool:
        """페이지 유효성 검사"""
        title = page.get('title', '')
        text = page.get('text', '')
        
        # 리다이렉트 페이지 제외
        if text.strip().startswith('#REDIRECT') or text.strip().startswith('#넘겨주기'):
            return False
        
        # 특수 페이지 제외
        if any(prefix in title for prefix in ['위키백과:', 'Wikipedia:', '파일:', 'File:', 
                                                '틀:', 'Template:', '분류:', 'Category:',
                                                '사용자:', 'User:', '토론:', 'Talk:']):
            return False
        
        # 너무 짧은 페이지 제외
        if len(text) < 500:
            return False
        
        return True


class WikiTextCleaner:
    """위키 마크업 정제"""
    
    def __init__(self):
        # 정규식 패턴들
        self.patterns = {
            # 위키 링크: [[링크|표시텍스트]] → 표시텍스트
            'wiki_link': re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]'),
            
            # 외부 링크: [http://... 텍스트] → 텍스트
            'external_link': re.compile(r'\[https?://[^\s\]]+ ([^\]]+)\]'),
            
            # 참조: <ref>...</ref> → 제거
            'ref_tag': re.compile(r'<ref[^>]*>.*?</ref>', re.DOTALL),
            'ref_self_closing': re.compile(r'<ref[^>]*/>', re.DOTALL),
            
            # HTML 태그 제거
            'html_tag': re.compile(r'<[^>]+>'),
            
            # 파일/이미지: [[파일:...]] → 제거
            'file': re.compile(r'\[\[(?:파일|File|그림|Image):[^\]]+\]\]', re.IGNORECASE),
            
            # 틀: {{...}} → 제거 (중첩 처리)
            'template': re.compile(r'\{\{[^\}]+\}\}'),
            
            # 테이블: {| ... |} → 제거
            'table': re.compile(r'\{\|.*?\|\}', re.DOTALL),
            
            # 분류: [[분류:...]] → 제거
            'category': re.compile(r'\[\[(?:분류|Category):[^\]]+\]\]', re.IGNORECASE),
            
            # 주석: <!-- ... --> → 제거
            'comment': re.compile(r'<!--.*?-->', re.DOTALL),
            
            # 마크업: '''굵게''', ''기울임'' → 텍스트만
            'bold': re.compile(r"'''([^']+)'''"),
            'italic': re.compile(r"''([^']+)''"),
            
            # 등호 헤더: == 제목 == → 제목
            'header': re.compile(r'^=+\s*(.+?)\s*=+$', re.MULTILINE),
            
            # 목록: *, #, : 등 → 제거
            'list': re.compile(r'^[\*\#\:\;]+\s*', re.MULTILINE),
        }
    
    def clean(self, text: str) -> str:
        """위키 마크업 정제"""
        
        # 1. 주석 제거
        text = self.patterns['comment'].sub('', text)
        
        # 2. 참조 제거
        text = self.patterns['ref_tag'].sub('', text)
        text = self.patterns['ref_self_closing'].sub('', text)
        
        # 3. 파일/이미지 제거
        text = self.patterns['file'].sub('', text)
        
        # 4. 테이블 제거
        text = self.patterns['table'].sub('', text)
        
        # 5. 틀 제거 (여러 번 반복 - 중첩 처리)
        for _ in range(3):
            text = self.patterns['template'].sub('', text)
        
        # 6. 분류 제거
        text = self.patterns['category'].sub('', text)
        
        # 7. 링크 처리
        text = self.patterns['wiki_link'].sub(r'\1', text)
        text = self.patterns['external_link'].sub(r'\1', text)
        
        # 8. 마크업 제거
        text = self.patterns['bold'].sub(r'\1', text)
        text = self.patterns['italic'].sub(r'\1', text)
        
        # 9. 헤더 처리
        text = self.patterns['header'].sub(r'\1', text)
        
        # 10. 목록 기호 제거
        text = self.patterns['list'].sub('', text)
        
        # 11. HTML 태그 제거
        text = self.patterns['html_tag'].sub('', text)
        
        # 12. 과도한 공백 정리
        text = re.sub(r'\n\n\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()


def process_wiki_dump(dump_path: str, 
                     output_path: str,
                     min_chunk_length: int = 500,
                     max_chunk_length: int = 1000):
    """
    위키 덤프 → JSONL 변환
    
    Args:
        dump_path: 입력 덤프 파일
        output_path: 출력 JSONL 파일
        min_chunk_length: 최소 청크 길이
        max_chunk_length: 최대 청크 길이
    """
    logger.info("="*60)
    logger.info("🔄 Wikipedia Dump Processing Pipeline")
    logger.info("="*60)
    logger.info(f"Input:  {dump_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Chunk:  {min_chunk_length}-{max_chunk_length} chars")
    logger.info("="*60 + "\n")
    
    # 파서 및 정제기
    parser = WikiDumpParser()
    cleaner = WikiTextCleaner()
    
    # Chunker (기존 wiki_reparser.py에서 import)
    try:
        from wiki_reparser import WikiChunker
    except ImportError:
        logger.error("❌ wiki_reparser.py not found in current directory!")
        logger.error("   Make sure wiki_reparser.py is in the same folder.")
        return
    
    chunker = WikiChunker(min_length=min_chunk_length, max_length=max_chunk_length)
    
    # 출력 파일
    output_file = open(output_path, 'w', encoding='utf-8')
    
    chunk_id = 0
    
    try:
        # 덤프 파싱
        for page in parser.parse_xml_dump(dump_path):
            title = page['title']
            raw_text = page['text']
            
            # 위키 마크업 정제
            clean_text = cleaner.clean(raw_text)
            
            if len(clean_text) < 200:
                continue
            
            # 청크 생성
            chunks = chunker.create_chunks(clean_text, doc_id=f"wiki_{title}")
            
            # 저장
            for chunk in chunks:
                chunk['title'] = title
                chunk['chunk_id'] = chunk_id
                
                output_file.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                chunk_id += 1
            
            parser.stats['total_chunks'] += len(chunks)
            
            # 진행 상황 (1000개마다)
            if parser.stats['valid_pages'] % 1000 == 0:
                logger.info(f"  Progress: {parser.stats['valid_pages']} pages, "
                          f"{parser.stats['total_chunks']} chunks")
    
    finally:
        output_file.close()
    
    # 최종 통계
    logger.info("\n" + "="*60)
    logger.info("📊 Processing Complete")
    logger.info("="*60)
    logger.info(f"  Total Pages:   {parser.stats['total_pages']:,}")
    logger.info(f"  Valid Pages:   {parser.stats['valid_pages']:,}")
    logger.info(f"  Skipped Pages: {parser.stats['skipped_pages']:,}")
    logger.info(f"  Total Chunks:  {parser.stats['total_chunks']:,}")
    logger.info(f"  Output:        {output_path}")
    logger.info("="*60 + "\n")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='위키피디아 덤프 직접 파싱',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 한국어 위키 덤프 처리
  python wiki_dump_parser.py -i kowiki-latest-pages-articles.xml.bz2 -o wiki_processed.jsonl
  
  # 압축 해제된 XML 처리
  python wiki_dump_parser.py -i kowiki-latest-pages-articles.xml -o wiki_processed.jsonl
  
  # 청크 크기 조정
  python wiki_dump_parser.py -i dump.xml.bz2 -o output.jsonl --min-length 300 --max-length 800

Download Korean Wikipedia dump:
  wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
        """
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='입력 덤프 파일 (.xml.bz2 또는 .xml)')
    parser.add_argument('--output', '-o', required=True,
                       help='출력 JSONL 파일')
    parser.add_argument('--min-length', type=int, default=500,
                       help='최소 청크 길이 (기본: 500)')
    parser.add_argument('--max-length', type=int, default=1000,
                       help='최대 청크 길이 (기본: 1000)')
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        logger.error(f"❌ Input file not found: {args.input}")
        logger.info("\n💡 Download Korean Wikipedia dump:")
        logger.info("   wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2")
        return
    
    # 처리
    process_wiki_dump(
        dump_path=args.input,
        output_path=args.output,
        min_chunk_length=args.min_length,
        max_chunk_length=args.max_length
    )
    
    logger.info("✅ All done!\n")


if __name__ == "__main__":
    main()