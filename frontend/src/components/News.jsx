import React, { useEffect, useState } from "react";
import { getNews } from "../services/api";
import { format } from "date-fns";
import { ExternalLink, Tag, Calendar, Globe, User } from "lucide-react";

const sentimentColor = (sentiment) => {
  switch (sentiment) {
    case "POSITIVE":
      return "text-green-400 border-green-400/30 bg-green-400/10";
    case "NEGATIVE":
      return "text-red-400 border-red-400/30 bg-red-400/10";
    default:
      return "text-slate-400 border-slate-400/30 bg-slate-400/10";
  }
};

const News = ({ startTime, endTime, shouldLoad }) => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [error, setError] = useState(null);

  // Limit per page
  const LIMIT = 10;

  // Fetch news when shouldLoad trigger fires
  useEffect(() => {
    if (!shouldLoad || !startTime || !endTime) return;

    setNews([]);
    setPage(1);
    setTotalPages(0);
    setError(null);
    fetchNews(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shouldLoad, startTime, endTime]);

  const fetchNews = async (pageNum) => {
    setLoading(true);
    setError(null);
    try {
      const startDate = new Date(startTime * 1000).toISOString();
      const endDate = new Date(endTime * 1000).toISOString();

      console.log("Fetching news with:", {
        startTime,
        endTime,
        startDate,
        endDate,
        page: pageNum,
        limit: LIMIT,
      });

      const resp = await getNews(null, startDate, endDate, pageNum, LIMIT);

      // Handle the new response structure
      if (resp && resp.data && Array.isArray(resp.data)) {
        console.log("News response:", {
          count: resp.data.length,
          total: resp.pagination?.total,
          totalPages: resp.pagination?.totalPages,
        });
        setNews(resp.data);
        setTotalPages(resp.pagination?.totalPages || 0);
        setPage(pageNum);
      } else if (resp && resp.pagination) {
        // Valid response but empty data
        setNews([]);
        setTotalPages(resp.pagination.totalPages || 0);
        setPage(pageNum);
      } else {
        // Unexpected response format
        console.warn("Unexpected news response format:", resp);
        setNews([]);
        setTotalPages(0);
        setError("Unexpected response format");
      }
    } catch (error) {
      console.error("Failed to fetch news", error);
      setNews([]);
      setTotalPages(0);
      setError(error.message || "Failed to load news");
    } finally {
      setLoading(false);
    }
  };

  const getPageNumbers = () => {
    const pages = [];
    const maxVisible = 5;

    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      let start = Math.max(1, page - 2);
      let end = Math.min(totalPages, page + 2);

      if (page <= 3) {
        end = Math.min(5, totalPages);
      } else if (page >= totalPages - 2) {
        start = Math.max(1, totalPages - 4);
      }

      for (let i = start; i <= end; i++) pages.push(i);
    }
    return pages;
  };

  return (
    <div className="h-full flex flex-col p-4">
      <h2 className="text-xl font-bold mb-4 sticky top-0 bg-slate-900 py-2 z-10 w-full shrink-0">
        Crypto News
      </h2>

      <div className="flex-1 overflow-y-auto space-y-4 pr-2">
        {error && !loading && (
          <div className="text-center py-10">
            <p className="text-red-400 mb-2">⚠️ Error loading news</p>
            <p className="text-slate-500 text-sm">{error}</p>
            <button
              onClick={() => fetchNews(page)}
              className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {!error && news.length === 0 && !loading && (
          <div className="text-center text-slate-500 py-10">
            <p className="mb-2">📰 No news found in this range.</p>
            <p className="text-xs text-slate-600">
              Try selecting a different date range.
            </p>
          </div>
        )}

        {!error &&
          news.map((item, index) => (
            <div
              key={`${item.guid || item.id}-${index}`}
              className="bg-slate-800 rounded-lg border border-slate-700 hover:border-blue-500 transition-all overflow-hidden flex flex-col md:flex-row gap-4"
            >
              {/* Image Section */}
              {item.imageUrl && (
                <div className="w-full md:w-32 h-32 shrink-0 bg-slate-900 relative">
                  <img
                    src={item.imageUrl}
                    alt={item.title}
                    className="w-full h-full object-cover opacity-80 hover:opacity-100 transition-opacity"
                    onError={(e) => (e.target.style.display = "none")}
                  />
                </div>
              )}

              {/* Content Section */}
              <div
                className={`flex-1 p-3 md:pl-0 ${!item.imageUrl ? "md:p-3" : ""}`}
              >
                <div className="flex justify-between items-start gap-2 mb-1">
                  <a
                    href={item.url || "#"}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-bold text-blue-400 hover:underline hover:text-blue-300 flex items-center gap-1 leading-tight"
                  >
                    {item.title || "Untitled"}
                    <ExternalLink size={12} />
                  </a>
                  {item.sentiment && (
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${sentimentColor(item.sentiment)}`}
                    >
                      {item.sentiment}
                    </span>
                  )}
                </div>

                {item.subtitle && (
                  <p className="text-xs text-slate-300 mb-2 line-clamp-2">
                    {item.subtitle}
                  </p>
                )}

                <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400 mb-2">
                  <div className="flex items-center gap-1">
                    <Globe size={10} />
                    <span className="text-slate-300">
                      {item.sourceName || "Unknown Source"}
                    </span>
                  </div>
                  {item.publishedOn && (
                    <div className="flex items-center gap-1">
                      <Calendar size={10} />
                      <span>
                        {format(
                          new Date(item.publishedOn * 1000),
                          "yyyy MMM dd, HH:mm",
                        )}
                      </span>
                    </div>
                  )}
                  {item.authors &&
                    (Array.isArray(item.authors)
                      ? item.authors.length > 0
                      : true) && (
                      <div className="flex items-center gap-1 truncate max-w-[150px]">
                        <User size={10} />
                        <span>
                          {Array.isArray(item.authors)
                            ? item.authors.join(", ")
                            : item.authors}
                        </span>
                      </div>
                    )}
                </div>

                {/* Tags/Keywords */}
                <div className="flex flex-wrap gap-1 mt-auto">
                  {item.categories &&
                    Array.isArray(item.categories) &&
                    item.categories.length > 0 &&
                    item.categories.slice(0, 3).map(
                      (cat, i) =>
                        cat && (
                          <span
                            key={i}
                            className="text-[10px] px-1.5 py-0.5 bg-slate-700 text-slate-300 rounded border border-slate-600"
                          >
                            {cat}
                          </span>
                        ),
                    )}
                  {(!item.categories ||
                    !Array.isArray(item.categories) ||
                    item.categories.length === 0) &&
                    item.keywords &&
                    typeof item.keywords === "string" &&
                    item.keywords.trim() && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-slate-700 text-slate-300 rounded border border-slate-600 flex items-center gap-1">
                        <Tag size={10} />
                        {item.keywords.slice(0, 30)}
                        {item.keywords.length > 30 && "..."}
                      </span>
                    )}
                </div>
              </div>
            </div>
          ))}

        {loading && (
          <div className="text-center p-4">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <p className="text-slate-400 mt-2">Loading news...</p>
          </div>
        )}
      </div>

      {/* Numbered Pagination */}
      {!error && totalPages > 1 && (
        <div className="pt-4 border-t border-slate-800 flex justify-center items-center gap-1 shrink-0">
          <button
            disabled={page === 1 || loading}
            onClick={() => fetchNews(page - 1)}
            className="px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-400 disabled:opacity-50 hover:bg-slate-700 text-xs"
          >
            Prev
          </button>

          {getPageNumbers().map((p) => (
            <button
              key={p}
              onClick={() => fetchNews(p)}
              disabled={loading}
              className={`w-7 h-7 flex items-center justify-center rounded text-xs font-bold transition-colors border ${
                page === p
                  ? "bg-blue-600 border-blue-500 text-white"
                  : "bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {p}
            </button>
          ))}

          <button
            disabled={page === totalPages || loading}
            onClick={() => fetchNews(page + 1)}
            className="px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-400 disabled:opacity-50 hover:bg-slate-700 text-xs"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

export default News;
