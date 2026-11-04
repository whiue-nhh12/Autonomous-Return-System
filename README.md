# Automous Return System

Hệ thống cung cấp các thành phần để thu thập chính sách trả hàng/hoàn tiền, nạp tài liệu vào RAG, tìm kiếm nội dung liên quan và tùy chọn trích xuất tri thức dạng đồ thị. Các thành phần hiện được tổ chức dưới dạng module Python; repository chưa có API hoặc giao diện người dùng tích hợp toàn bộ luồng thành một ứng dụng chạy độc lập.

## Tổng quan kiến trúc

```mermaid
flowchart TD
    subgraph Collect["Thu thập chính sách"]
        Shopee["Shopee"] --> Firecrawl["Firecrawl"]
        TikTok["TikTok Shop"] --> BS4["HTTP + BeautifulSoup"]
        Firecrawl --> Markdown["Markdown + crawl manifest"]
        BS4 --> Markdown
    end

    Markdown -. "chọn file/URL để nạp" .-> Validate

    subgraph Ingest["Nạp và publish tài liệu"]
        Input["File local hoặc URL"] --> Validate["Validation"]
        Validate --> Docling["Docling parse"]
        Docling --> VLM["VLM bổ sung hình ảnh"]
        VLM --> Parsed["ParsedDocument"]
        Parsed --> Clean["Cleaning"]
        Clean --> Chunk["Chunking theo heading"]
        Chunk --> Embed["Tạo embeddings"]
        Embed --> Postgres[("PostgreSQL<br/>trạng thái + phiên bản")]
        Embed --> Qdrant[("Qdrant<br/>vector index")]
        Embed --> ES[("Elasticsearch<br/>keyword index")]
        Postgres --> Published["Publish phiên bản"]
    end

    subgraph Retrieve["Truy vấn RAG"]
        Question["Câu hỏi"] --> Prepare["Rewrite / HyDE<br/>(tùy chọn)"]
        Prepare --> Hybrid["Hybrid search + RRF"]
        Qdrant --> Hybrid
        ES --> Hybrid
        Published --> Filter["Lọc phiên bản publish"]
        Hybrid --> Filter
        Filter --> Rerank["Rerank<br/>(tùy chọn)"]
        Rerank --> Results["Các đoạn liên quan<br/>(chưa sinh câu trả lời)"]
    end

    Clean -. "gọi riêng, tùy chọn" .-> Classify
    subgraph Graph["GraphRAG tùy chọn"]
        Classify["Phân loại block"] --> Extract["Trích xuất entity<br/>và policy rule"]
        Extract --> Normalize["Chuẩn hóa + liên kết"]
        Normalize --> Outbox[("PostgreSQL outbox")]
        Outbox --> Neo4j[("Neo4j graph")]
    end

    classDef source fill:#eaf2ff,stroke:#4b78c2,color:#17345c
    classDef process fill:#f3edff,stroke:#8564c2,color:#39245f
    classDef store fill:#e7f7ef,stroke:#41966a,color:#174d31
    classDef result fill:#fff3d9,stroke:#d69b2e,color:#65460a
    class Shopee,TikTok,Input,Question source
    class Firecrawl,BS4,Validate,Docling,VLM,Parsed,Clean,Chunk,Embed,Prepare,Hybrid,Filter,Rerank,Classify,Extract,Normalize process
    class Markdown,Postgres,Qdrant,ES,Outbox,Neo4j store
    class Published,Results result
```

## Luồng hoạt động

Các module trong repository là những khối xử lý có thể kết hợp, chưa phải một ứng dụng web end-to-end. Không có API/UI tự động chuyển kết quả crawl sang ingestion hoặc tạo câu trả lời hội thoại. Luồng tích hợp dự kiến như sau:

### 1. Cập nhật nguồn chính sách

```text
Trang chính sách
  → chạy crawler phù hợp
  → file Markdown + manifest.json
  → chọn file/URL để đưa vào quy trình ingestion
```

Crawler Firecrawl tìm kiếm và scrape các trang thuộc domain được cấu hình; crawler TikTok Shop tải và trích xuất bài viết chính thức bằng BeautifulSoup. Hai crawler lưu Markdown và manifest crawl, nhưng không tự gọi ingestion hoặc cập nhật các chỉ mục tìm kiếm.

### 2. Nạp và publish tài liệu

```text
File local hoặc URL
  → kiểm tra nguồn, định dạng và kích thước
  → Docling parse thành ParsedDocument
  → VLM bổ sung nội dung hình ảnh (nếu có/cấu hình)
  → cleaning
  → chunking theo heading/cấu trúc
  → tạo embedding
  → ghi phiên bản vào PostgreSQL và chỉ mục vào Qdrant/Elasticsearch
  → đánh dấu READY rồi publish phiên bản
```

`IngestionPipeline` hiện chỉ thực hiện validation → parse → VLM và trả về `ParsedDocument`. Cleaning, chunking, embedding và lưu trữ là các module riêng; ứng dụng tích hợp cần gọi chúng trong callback xử lý của `DocumentVersionService`. Service tạo phiên bản mới, bỏ qua nội dung không đổi theo content hash, chỉ publish sau khi xử lý và ghi chỉ mục thành công; nếu xử lý thất bại, phiên bản bị đánh dấu lỗi và dữ liệu chỉ mục tương ứng được dọn dẹp. PostgreSQL là nơi lưu trạng thái và phiên bản tài liệu; manifest `published_versions.json` là cache giúp retrieval lọc đúng phiên bản hiện hành.

### 3. Tìm kiếm nội dung

```text
Câu hỏi
  → (tùy chọn) viết lại câu hỏi / tạo HyDE
  → tìm kiếm từ khóa trên Elasticsearch và vector trên Qdrant
  → kết hợp thứ hạng bằng Reciprocal Rank Fusion (RRF)
  → lọc theo phiên bản đã publish
  → (tùy chọn) rerank
  → trả về các đoạn liên quan và điểm số
```

`RetrievalPipeline.search()` trả về các kết quả truy xuất, không tự sinh câu trả lời cuối cùng cho người dùng. Query rewriting, HyDE và reranking là tùy chọn; mặc định tắt.

### 4. Trích xuất tri thức đồ thị (tùy chọn)

Sau khi có tài liệu đã cleaning và số phiên bản, ứng dụng có thể gọi riêng GraphRAG:

```text
Cleaned_Document
  → phân loại block
  → trích xuất entity và policy rule từ block phù hợp
  → chuẩn hóa entity, liên kết rule
  → lưu chunk/outbox trong PostgreSQL
  → đồng bộ node và quan hệ sang Neo4j
```

GraphRAG không nằm trong `IngestionPipeline`. PostgreSQL giữ nội dung chunk và outbox để có thể thử đồng bộ lại; Neo4j giữ cấu trúc đồ thị và tham chiếu chunk.

## Các thành phần

### Thu thập chính sách

Hai script crawl độc lập lưu nội dung Markdown và lịch sử vào thư mục
`rag/data/tool/return_refund/`:

| Script | Nguồn | Cách lấy dữ liệu |
| --- | --- | --- |
| `rag/data/tool/crawl_data.py` | Các nguồn được khai báo trong `POLICY_SOURCES` | Firecrawl Search rồi scrape các URL kết quả thuộc domain được cho phép |
| `rag/data/tool/crawl_tiktokshop.py` | Bài chính sách TikTok Shop đã cấu hình trong script | GET trang chính thức, dùng BeautifulSoup lấy tiêu đề và phần `.converted-html` |

Lưu ý: `POLICY_SOURCES` hiện có một URL chia sẻ ChatGPT bên cạnh nguồn Shopee và TikTok Shop. Nếu dữ liệu chỉ được phép lấy từ nguồn chính thức, hãy xem lại hoặc bỏ nguồn này trước khi chạy crawler Firecrawl.

Thư mục dữ liệu có:

- File `.md` chứa nội dung crawl. TikTok tạo tên file riêng theo thời điểm crawl; Firecrawl dùng tên dựa trên URL nên lần crawl lại cùng URL có thể ghi đè file Markdown trước đó, dù manifest vẫn được append bản ghi mới.
- `manifest.json`: một **JSON array** ghi lịch sử crawl, gồm `platform`, `tool`, `title`, `url`, `crawled_at` và `markdown_path`. Các crawler thêm bản ghi mới vào cuối danh sách nếu manifest đã tồn tại.
- Giá trị `tool` cho biết nguồn tạo tài liệu: `firecrawl` hoặc `beautifulsoup`.

Manifest crawl này **khác** với `published_versions.json`: manifest crawl mô tả các file chính sách đã tải; `published_versions.json` là cache ánh xạ `doc_id` sang phiên bản đã publish để retrieval lọc đúng phiên bản. Không dùng hai file thay thế cho nhau.

Chạy từ thư mục gốc repository:

```powershell
# Thu thập các nguồn trong POLICY_SOURCES; cần FIRECRAWL_API_KEY
python -m rag.data.tool.crawl_data --limit 7

# Thu thập bài TikTok Shop bằng BeautifulSoup
python -m rag.data.tool.crawl_tiktokshop
```

Có thể truyền thư mục đầu ra riêng bằng `--output-dir`, ví dụ:

```powershell
python -m rag.data.tool.crawl_tiktokshop --output-dir .\data\return_refund
```

### Ingestion và lưu trữ

- **Validation** (`rag/ingestion/validator.py`): nhận file local hoặc URL; giới hạn định dạng và kích thước theo settings.
- **Parsing** (`rag/ingestion/parser.py`): Docling chuyển tài liệu thành các `ContentBlock`, giữ cấu trúc heading.
- **VLM** (`rag/ingestion/vlm/`): chỉ xử lý hình ảnh trong tài liệu nếu có block hình phù hợp; provider được cấu hình trong ingestion settings.
- **Cleaning** (`rag/cleaning/cleaner.py`): lọc nội dung không cần thiết và chuẩn hóa văn bản. Đây là module riêng, chưa được gọi bên trong `IngestionPipeline`.
- **Chunking** (`rag/chunking/chunker.py`): chia nội dung theo heading/list và giới hạn kích thước chunk.
- **Embedding** (`rag/embedding/embedding.py`): tạo vector bằng `BAAI/bge-m3`, vector 1024 chiều.
- **Versioning** (`rag/document_store/document/service.py`): PostgreSQL lưu tài liệu và các phiên bản; chỉ publish phiên bản đã xử lý thành công. Khi cập nhật nội dung giống phiên bản hiện tại, service trả về no-op.
- **Chỉ mục truy vấn**: Qdrant lưu vector; Elasticsearch phục vụ tìm kiếm từ khóa. Các adapter hỗ trợ ghi và xóa theo phiên bản.

`IngestionPipeline` điều phối Validation → Docling parse → VLM và trả về `ParsedDocument`. Script [`scripts/ingest_to_storage.py`](./scripts/ingest_to_storage.py) nối pipeline này với cleaning, chunking, embedding, versioning PostgreSQL và các index Qdrant/Elasticsearch cho file local.

Có thể chạy pipeline tích hợp cho một file local từ thư mục gốc repository:

```powershell
python -m scripts.ingest_to_storage .\rag\data\tool\return_refund
```

Khi truyền thư mục, script tìm đệ quy các file có định dạng được hỗ trợ, bỏ qua `manifest.json`, xử lý từng file và tiếp tục nếu một file lỗi; kết quả cuối cùng trả mã lỗi nếu có file thất bại. ID tài liệu được tạo ổn định theo đường dẫn tương đối trong thư mục, nên chạy lại sẽ cập nhật phiên bản của từng file thay vì tạo bản ghi mới. Cũng có thể truyền một file riêng; khi đó `--doc-id` dùng để cập nhật tài liệu đã có. Script tạo schema PostgreSQL nếu chưa có, nạp vector 1024 chiều vào Qdrant và chunk vào Elasticsearch, sau đó publish phiên bản. Nếu Qdrant đã có collection cũ với vector 768 chiều, cần xóa và tạo lại collection đó trước khi chạy lại; script sẽ báo lỗi cấu hình dimension ngay khi khởi động thay vì đợi đến lúc upload. Cần cài `sentence-transformers`, chạy các dịch vụ trong Docker Compose và cấu hình khóa VLM nếu tài liệu có ảnh cần trích xuất. `QDRANT_URL` có thể dùng để thay đổi URL Qdrant mặc định.

### Truy vấn RAG

`RetrievalPipeline` nhận câu hỏi và:

1. Đọc các phiên bản đã publish (ưu tiên manifest phiên bản; nếu không đọc được thì có thể tải lại từ PostgreSQL).
2. Tạo query embedding và đồng thời tìm kiếm từ khóa/vector qua Elasticsearch và Qdrant.
3. Kết hợp thứ hạng bằng Reciprocal Rank Fusion (RRF), chỉ giữ kết quả thuộc phiên bản đang publish.
4. Tùy chọn áp dụng query rewriting, HyDE và reranker. Các tính năng này tắt mặc định và bật qua retrieval settings.
5. Trả danh sách các đoạn liên quan cùng điểm số; API/giao diện hội thoại hoàn chỉnh chưa được nối vào repository.

Chạy thử một câu hỏi end-to-end bằng `python -m scripts.query "Câu hỏi của bạn"`. Kết quả console hiển thị thời gian khởi tạo, chuẩn bị query, retrieval, dựng context và sinh câu trả lời; `--json` trả các số liệu tổng hợp trong trường `timings_ms` và breakdown các bước retrieval trong `retrieval_timings_ms`. Tổng thời gian được tính từ lúc bắt đầu khởi tạo đến khi sinh xong câu trả lời, không bao gồm đóng kết nối.

Đo chi tiết thời gian từ input đến output theo giai đoạn, module và từng hàm Python (bao gồm các hàm tìm kiếm chạy song song):

```powershell
python -m scripts.measure "Câu hỏi của bạn" --limit 5
```

Embedding model được nạp trước khi bắt đầu đo; nếu bật reranker, model reranker cũng được nạp trước. Thời gian inference, truy xuất và sinh câu trả lời vẫn được tính. Script tách riêng embedding, Elasticsearch, Qdrant, RRF, rewrite, HyDE, reranker và GraphRAG nếu các thành phần đó được bật; các bước được sắp theo thời gian giảm dần. Elasticsearch và Qdrant chạy song song nên thời gian hai backend có thể chồng lấp, không cộng chúng để tính wall time. Mặc định hiển thị toàn bộ hàm; dùng `--top 50` để giới hạn số dòng. Thời gian function từ các thread song song được cộng theo thread nên có thể lớn hơn tổng wall-clock.

### GraphRAG tùy chọn

`rag/graph/` có pipeline phân loại block phù hợp, trích xuất và chuẩn hóa entity/policy rule, liên kết rule và đồng bộ dữ liệu qua PostgreSQL outbox sang Neo4j. `scripts/ingest_to_storage.py` dùng Groq để phân loại và trích xuất các block phù hợp; `IngestionPipeline` cấp thấp vẫn không tự gọi GraphRAG. Cấu hình `GROQ_API_KEY` cùng kết nối Neo4j trước khi chạy script. Xem thêm [`rag/graph/README.md`](./rag/graph/README.md).

## Chạy dịch vụ phụ trợ

`docker-compose.yaml` định nghĩa PostgreSQL, Qdrant và Elasticsearch cho môi trường phát triển:

```powershell
docker compose up -d
```

Các cổng mặc định được mở lần lượt là PostgreSQL `5432`, Qdrant `6333`/`6334`, Elasticsearch `9200`. Thông tin đăng nhập trong Compose chỉ phù hợp cho môi trường local; hãy cấu hình secrets riêng trước khi triển khai ra môi trường dùng chung hoặc production. Neo4j không nằm trong Compose file hiện tại và chỉ cần khi bật GraphRAG.

## Backend database và repository

`backend/db/` quản lý SQLAlchemy engine/session cho các model nghiệp vụ trong `backend/models/`. Tầng này đọc `DATABASE_URL` từ môi trường; với PostgreSQL trong Docker Compose local:

```text
DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag
```

`get_db()` cung cấp session có vòng đời theo request và không tự commit. Repository trong `backend/repositories/` nhận session qua constructor, thực hiện `flush()` để trả dữ liệu đã được ghi trong transaction hiện tại; service hoặc endpoint chịu trách nhiệm commit/rollback, ví dụ bằng `with session.begin():`. Có repository CRUD cho từng model, cùng các lookup theo email, SKU và mã đơn/yêu cầu trả.

`init_db()` tạo các bảng còn thiếu để phát triển hoặc kiểm thử; thay đổi schema production cần dùng migration thay vì dựa vào `create_all()`. Chưa có API/service nghiệp vụ tích hợp các repository này.

## Agent state

`agent/agents/policy/graph.py` triển khai agent bằng LangGraph; worker retrieval và state cũng nằm trong `agent/agents/policy/`. Luồng hiện tại là `classify → retrieve_policy → answer_with_evidence`, nhánh truy xuất lỗi/không có kết quả đi qua `no_evidence`, còn câu hỏi ngoài phạm vi đi qua `clarify`. Node retrieval dùng `RetrievalPipeline`, áp dụng bộ lọc `doc_ids`, `policy_version` và `as_of_date` khi metadata có ngày hiệu lực/ngày publish. Node trả lời trích đoạn kèm nguồn; graph không tự phê duyệt hay từ chối yêu cầu trả hàng. Tạo graph bằng `PolicyAgentGraph(retrieval_pipeline)` rồi gọi `invoke("Tôi có thể đổi trả sản phẩm không?")`. Cài dependency từ `requirements.txt`. Pipeline hiện chỉ truy xuất phiên bản đã publish, nên yêu cầu `published_only=False` trả trạng thái `failed`.

Node `clarify` kết thúc request hiện tại. Khi người dùng gửi câu trả lời, gọi `invoke()` lần nữa với message mới và cùng `conversation_id`; graph chạy lại từ `START` với `request_id` mới. Không nối cạnh `clarify → START`, vì graph sẽ lặp lại mà chưa có input mới.

`agent/core/state/shared.py` định nghĩa state dùng chung giữa các agent. `AgentSharedState` là contract để orchestrator ghép kết quả; các worker nên trả kết quả riêng, không cùng sửa một state mutable khi chạy song song. `agent/agents/policy/state/` chỉ chứa state cục bộ cho retrieval: `DocumentRetrievalLocalState` giữ query, filter, candidates và evidence tạm; chỉ các citation được chọn chuyển thành `PolicyEvidence` trong shared state. Các state này là dữ liệu runtime, không thay thế PostgreSQL hay lịch sử hội thoại bền vững.

## Cấu hình

Ingestion đọc cấu hình từ `.env` ở thư mục gốc hoặc biến môi trường. Các nhóm cấu hình chính:

- `VLM_SETTINGS__...`: provider, model và credentials cho VLM.
- `STORAGE_SETTINGS__...`: PostgreSQL, Elasticsearch, index và đường dẫn manifest phiên bản.
- `GRAPH_SETTINGS__...`: kết nối Neo4j và thiết lập GraphRAG.
- `GROQ_API_KEY`: khóa API Groq dùng cho phân loại và trích xuất GraphRAG.
- `GRAPH_SETTINGS__MODEL`: model Groq dùng để trích xuất; mặc định `openai/gpt-oss-120b`.
- `GRAPH_SETTINGS__CLASSIFIER_MODEL`: model Groq dùng để phân loại; mặc định `qwen/qwen3.8-27b`.
- `RETRIEVAL_...`: bật/tắt rewriting, HyDE, reranking và cấu hình retrieval.
- `FIRECRAWL_API_KEY`: cần cho `crawl_data.py`.

Không commit API key, mật khẩu hoặc file `.env` có secrets. Đặt giá trị credentials trong môi trường local/deployment của bạn.

## Cài đặt và kiểm thử

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-postgres.txt
python -m pip install sentence-transformers
python -m unittest discover -s test -v
```

`requirements-postgres.txt` cài PostgreSQL driver Psycopg cần cho URL `postgresql+psycopg`. Một số backend/model cần dịch vụ ngoài hoặc credentials tương ứng. Bài test với mock có thể chạy mà không cần gọi API crawler thực tế.

## Bố cục mã nguồn

```text
rag/
  ingestion/       Kiểm tra nguồn, parse tài liệu, VLM
  cleaning/        Làm sạch các block đã parse
  chunking/        Chia tài liệu thành các đoạn
  embedding/       Sinh vector cho chunk
  document_store/  PostgreSQL repository và version service
  vector_db/       Qdrant adapter
  search/          Elasticsearch và tìm kiếm từ khóa
  retrieval/       Hybrid search, query augmentation, reranking
  graph/           GraphRAG và tích hợp Neo4j
  data/tool/       Các script crawl chính sách
backend/
  db/              SQLAlchemy engine, session và khởi tạo schema
  models/          Model SQLAlchemy cho backend trả hàng
  repositories/    Truy vấn và CRUD theo model
agent/
  agents/policy/   Policy retrieval worker, graph và state cục bộ
  orchestration/   Manager graph, routing và node handlers
scripts/           CLI điều phối ingestion đến storage
test/              Unit test
```
