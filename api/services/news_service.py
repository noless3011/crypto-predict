from datetime import datetime
from typing import List, Optional
import clickhouse_connect
from api.config import settings
from api.models.schemas import NewsItem

class NewsService:
    def _get_client(self):
        return clickhouse_connect.get_client(
            host=settings.CLICKHOUSE_HOST,
            port=settings.CLICKHOUSE_PORT,
            username=settings.CLICKHOUSE_USER,
            password=settings.CLICKHOUSE_PASSWORD,
            database=settings.CLICKHOUSE_DB,
        )

    def get_news(self, symbol: str, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: int = 20) -> List[NewsItem]:
        with self._get_client() as client:
            conditions = []
            if start_time:
                conditions.append(f"publishedOn >= {start_time}")
            if end_time:
                conditions.append(f"publishedOn <= {end_time}")
                
            # Filter by symbol in keywords if possible
            # User request: "no need to input ticker name, just the date time range"
            # if symbol:
            #    conditions.append(f"keywords LIKE '%{symbol}%'")

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            # Using columns identified from user error message:
            # id, type, guid, url, publishedOn, imageUrl, title, subtitle, authors, sourceId, rawBody, keywords, sentiment, sourceName, categories, htmlBody, version
            
            query = f"""
            SELECT publishedOn, title, rawBody, sourceName, url, sentiment 
            FROM news
            {where_clause}
            ORDER BY publishedOn DESC
            LIMIT {limit}
            """
            
            try:
                result = client.query(query)
                news_items = []
                for row in result.result_rows:
                    published_on = row[0]
                    title = row[1]
                    summary = row[2][:200] + "..." if row[2] else "" # Truncate rawBody for summary
                    source = row[3]
                    url = row[4]
                    sentiment = row[5]
                    
                    news_items.append(NewsItem(
                        id=f"news_{int(published_on)}_{hash(title)}",
                        title=title,
                        summary=summary,
                        source=source,
                        url=url,
                        published_at=int(published_on),
                        sentiment=str(sentiment)
                    ))
                return news_items
                
            except Exception as e:
                print(f"Error fetching news: {e}")
                return []
