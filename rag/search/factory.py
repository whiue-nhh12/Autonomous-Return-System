"""Factories for configured keyword-search components."""

from rag.core.config import StorageSettings
from rag.search.base_keyword_search import BaseKeywordSearch
from rag.search.elasticsearch_index import ElasticsearchKeywordIndex
from rag.search.vietnamese_tokenizer import VietnameseTokenizer


def create_elasticsearch_keyword_index(
    settings: StorageSettings,
    *,
    client=None,
    tokenizer: VietnameseTokenizer | None = None,
) -> BaseKeywordSearch:
    index = ElasticsearchKeywordIndex(
        url=settings.elasticsearch_url,
        index_name=settings.elasticsearch_index,
        api_key=settings.elasticsearch_api_key.get_secret_value() if settings.elasticsearch_api_key else None,
        username=settings.elasticsearch_username.get_secret_value() if settings.elasticsearch_username else None,
        password=settings.elasticsearch_password.get_secret_value() if settings.elasticsearch_password else None,
        request_timeout=settings.elasticsearch_request_timeout,
        client=client,
        tokenizer=tokenizer,
    )
    index.create_collection()
    return index
