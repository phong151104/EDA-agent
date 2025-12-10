"""
Query Rewriter.

Analyzes user queries to extract entities, keywords, and intent
for retrieving relevant sub-graph from Neo4j.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import config
from .models import AnalyzedQuery, ExtractedEntity, QueryIntent

logger = logging.getLogger(__name__)


# Vietnamese keywords mapping to concepts/metrics
KEYWORD_MAPPINGS = {
    # Revenue/Money
    "doanh thu": ["revenue", "total_amount", "vnpay_total_amount"],
    "doanh số": ["revenue", "sales"],
    "tiền": ["amount", "money", "price"],
    "lợi nhuận": ["profit", "vnpay_seat_profit"],
    "chi phí": ["cost", "expense"],
    "giá": ["price", "amount"],
    
    # Volume
    "đơn hàng": ["order", "orders"],
    "đơn": ["order"],
    "vé": ["ticket", "seat", "number_of_seats"],
    "ghế": ["seat", "number_of_seats"],
    "số lượng": ["count", "quantity", "number"],
    
    # Time
    "ngày": ["date", "day", "created_date"],
    "tháng": ["month"],
    "năm": ["year"],
    "tuần": ["week"],
    "hôm nay": ["today"],
    "hôm qua": ["yesterday"],
    "quý": ["quarter"],
    
    # Entities
    "ngân hàng": ["bank", "bank_id"],
    "rạp": ["cinema", "vendor"],
    "phim": ["film", "movie"],
    "khách hàng": ["customer", "user"],
    "campaign": ["campaign", "dim_campaign"],
    "khuyến mại": ["promotion", "discount"],
    
    # Trends/Comparison
    "tăng": ["increase", "up", "growth"],
    "giảm": ["decrease", "down", "decline"],
    "so sánh": ["compare", "vs", "versus"],
    "xu hướng": ["trend"],
    "trung bình": ["average", "avg"],
    "tổng": ["total", "sum"],
}

# Intent detection patterns
INTENT_PATTERNS = {
    QueryIntent.DIAGNOSTIC: [
        r"tại sao",
        r"vì sao",
        r"nguyên nhân",
        r"lý do",
        r"why",
    ],
    QueryIntent.COMPARATIVE: [
        r"so sánh",
        r"so với",
        r"versus",
        r"vs\.?",
        r"khác nhau",
    ],
    QueryIntent.TREND: [
        r"xu hướng",
        r"biến động",
        r"thay đổi",
        r"trend",
        r"theo thời gian",
    ],
    QueryIntent.AGGREGATION: [
        r"tổng",
        r"trung bình",
        r"sum",
        r"avg",
        r"count",
        r"theo\s+\w+",
    ],
    QueryIntent.DETAIL: [
        r"chi tiết",
        r"thông tin",
        r"detail",
        r"list",
        r"danh sách",
    ],
}


class QueryRewriter:
    """
    Analyzes and rewrites user queries for EDA.
    
    Extracts:
    - Intent (what type of analysis)
    - Entities (tables, columns, metrics mentioned)
    - Keywords (for Neo4j search)
    - Time ranges
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize QueryRewriter.
        
        Args:
            use_llm: Whether to use LLM for analysis (more accurate but slower)
        """
        self.use_llm = use_llm
        self._llm: ChatOpenAI | None = None
    
    @property
    def llm(self) -> ChatOpenAI:
        """Get or create LLM client."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=config.openai.model,
                api_key=config.openai.api_key,
                temperature=0,
            )
        return self._llm
    
    async def analyze(self, query: str) -> AnalyzedQuery:
        """
        Analyze a user query.
        
        Args:
            query: Raw user query
            
        Returns:
            AnalyzedQuery with extracted information
        """
        logger.info(f"Analyzing query: {query}")
        
        # Step 1: Detect intent
        intent = self._detect_intent(query)
        
        # Step 2: Extract keywords using mappings
        keywords = self._extract_keywords(query)
        
        # Step 3: Extract entities (basic rule-based)
        entities = self._extract_entities(query)
        
        # Step 4: Use LLM for deeper analysis if enabled
        if self.use_llm:
            llm_result = await self._llm_analyze(query, intent, keywords, entities)
            # Merge LLM results
            keywords.extend(llm_result.get("keywords", []))
            keywords = list(set(keywords))
            
            for e in llm_result.get("entities", []):
                entities.append(ExtractedEntity(
                    text=e.get("text", ""),
                    entity_type=e.get("type", "concept"),
                    normalized_name=e.get("normalized"),
                    confidence=e.get("confidence", 0.8),
                ))
            
            if llm_result.get("intent"):
                try:
                    intent = QueryIntent(llm_result["intent"])
                except ValueError:
                    pass
        
        return AnalyzedQuery(
            original_query=query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            time_range=self._extract_time_range(query),
            rewritten_query=self._rewrite_query(query, intent),
        )
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent using patterns."""
        query_lower = query.lower()
        
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return QueryIntent.EXPLORATORY
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords using Vietnamese-English mappings."""
        keywords = []
        query_lower = query.lower()
        
        for vn_term, en_terms in KEYWORD_MAPPINGS.items():
            if vn_term in query_lower:
                keywords.extend(en_terms)
        
        return list(set(keywords))
    
    def _extract_entities(self, query: str) -> list[ExtractedEntity]:
        """Basic entity extraction using patterns."""
        entities = []
        query_lower = query.lower()
        
        # Extract time mentions
        time_patterns = [
            (r"tháng\s+(\d{1,2})", "time"),
            (r"năm\s+(\d{4})", "time"),
            (r"q(\d)", "time"),
            (r"quý\s+(\d)", "time"),
        ]
        
        for pattern, entity_type in time_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                entities.append(ExtractedEntity(
                    text=match,
                    entity_type=entity_type,
                ))
        
        # Extract known concepts from mappings
        for vn_term in KEYWORD_MAPPINGS:
            if vn_term in query_lower:
                entities.append(ExtractedEntity(
                    text=vn_term,
                    entity_type="concept",
                    normalized_name=KEYWORD_MAPPINGS[vn_term][0],
                ))
        
        return entities
    
    def _extract_time_range(self, query: str) -> dict[str, str] | None:
        """Extract time range from query if present."""
        # Basic patterns - can be extended
        patterns = [
            (r"từ\s+(\d{1,2}/\d{1,2}/\d{4})\s+đến\s+(\d{1,2}/\d{1,2}/\d{4})", 
             lambda m: {"start": m.group(1), "end": m.group(2)}),
            (r"tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
             lambda m: {"month": m.group(1), "year": m.group(2)}),
            (r"năm\s+(\d{4})",
             lambda m: {"year": m.group(1)}),
        ]
        
        query_lower = query.lower()
        for pattern, extractor in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return extractor(match)
        
        return None
    
    def _rewrite_query(self, query: str, intent: QueryIntent) -> str:
        """Rewrite query to be more specific for EDA."""
        # Add context based on intent
        prefix = ""
        if intent == QueryIntent.DIAGNOSTIC:
            prefix = "Phân tích nguyên nhân: "
        elif intent == QueryIntent.TREND:
            prefix = "Phân tích xu hướng: "
        elif intent == QueryIntent.COMPARATIVE:
            prefix = "So sánh: "
        
        return prefix + query if prefix else query
    
    async def _llm_analyze(
        self,
        query: str,
        current_intent: QueryIntent,
        current_keywords: list[str],
        current_entities: list[ExtractedEntity],
    ) -> dict[str, Any]:
        """Use LLM for deeper query analysis."""
        
        system_prompt = """Bạn là một chuyên gia phân tích dữ liệu. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng 
để xác định họ cần thông tin gì từ database.

Trích xuất:
1. intent: loại phân tích (exploratory, diagnostic, comparative, trend, aggregation, detail)
2. keywords: các từ khóa tiếng Anh liên quan đến schema (table names, column names, metrics)
3. entities: các thực thể được đề cập với loại (table, column, metric, concept, time, value)

Trả về JSON format:
{
    "intent": "diagnostic",
    "keywords": ["revenue", "orders", "bank"],
    "entities": [
        {"text": "doanh thu", "type": "metric", "normalized": "total_revenue", "confidence": 0.9},
        {"text": "ngân hàng", "type": "table", "normalized": "bank", "confidence": 0.95}
    ],
    "clarifications": ["Cần xác định khoảng thời gian cụ thể"]
}"""

        user_prompt = f"""Câu hỏi: "{query}"

Đã phát hiện:
- Intent hiện tại: {current_intent.value}
- Keywords: {current_keywords}

Hãy phân tích sâu hơn và trả về JSON."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            
            # Parse JSON from response
            import json
            content = response.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
        
        return {}


# Quick analysis without LLM
def quick_analyze(query: str) -> AnalyzedQuery:
    """
    Quick synchronous query analysis without LLM.
    
    Useful for fast keyword extraction.
    """
    rewriter = QueryRewriter(use_llm=False)
    
    intent = rewriter._detect_intent(query)
    keywords = rewriter._extract_keywords(query)
    entities = rewriter._extract_entities(query)
    
    return AnalyzedQuery(
        original_query=query,
        intent=intent,
        entities=entities,
        keywords=keywords,
        time_range=rewriter._extract_time_range(query),
    )
