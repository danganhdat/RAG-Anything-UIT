# BÁO CÁO CHI TIẾT HỆ THỐNG RAG-ANYTHING-UIT

## Hệ thống Truy xuất và Sinh câu trả lời Đa phương thức dựa trên Đồ thị Tri thức

---

**Nhóm thực hiện:** Nhóm RAG-Anything-UIT  
**Trường:** Đại học Công nghệ Thông tin — ĐHQG TP.HCM (UIT)  
**Ngày báo cáo:** Tháng 5/2026

---

## Mục lục

1. [Giới thiệu tổng quan](#1-giới-thiệu-tổng-quan)
2. [Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết)
   - 2.1 [Retrieval-Augmented Generation (RAG)](#21-retrieval-augmented-generation-rag)
   - 2.2 [LightRAG — RAG dựa trên Đồ thị Tri thức](#22-lightrag--rag-dựa-trên-đồ-thị-tri-thức)
   - 2.3 [RAG-Anything — Mở rộng đa phương thức](#23-rag-anything--mở-rộng-đa-phương-thức)
3. [Thiết kế hệ thống](#3-thiết-kế-hệ-thống)
   - 3.1 [Kiến trúc tổng thể](#31-kiến-trúc-tổng-thể)
   - 3.2 [Cấu trúc thư mục dự án](#32-cấu-trúc-thư-mục-dự-án)
   - 3.3 [Các thành phần chính](#33-các-thành-phần-chính)
4. [Pipeline nhập liệu tài liệu (Ingestion)](#4-pipeline-nhập-liệu-tài-liệu-ingestion)
   - 4.1 [Giai đoạn 1: Phân tích tài liệu (MinerU)](#41-giai-đoạn-1-phân-tích-tài-liệu-mineru)
   - 4.2 [Giai đoạn 2: Phân tích nội dung đa phương thức](#42-giai-đoạn-2-phân-tích-nội-dung-đa-phương-thức)
   - 4.3 [Giai đoạn 3: Xây dựng đồ thị tri thức](#43-giai-đoạn-3-xây-dựng-đồ-thị-tri-thức)
5. [Pipeline truy vấn (Query)](#5-pipeline-truy-vấn-query)
   - 5.1 [Trích xuất từ khóa](#51-trích-xuất-từ-khóa)
   - 5.2 [Các chế độ truy vấn](#52-các-chế-độ-truy-vấn)
   - 5.3 [Reranking](#53-reranking)
   - 5.4 [Sinh câu trả lời](#54-sinh-câu-trả-lời)
6. [Chi tiết triển khai](#6-chi-tiết-triển-khai)
   - 6.1 [Adapter Layer — Giao tiếp với LLM](#61-adapter-layer--giao-tiếp-với-llm)
   - 6.2 [Storage Layer — Lưu trữ dữ liệu](#62-storage-layer--lưu-trữ-dữ-liệu)
   - 6.3 [Service Layer — Logic nghiệp vụ](#63-service-layer--logic-nghiệp-vụ)
   - 6.4 [API Layer — FastAPI Backend](#64-api-layer--fastapi-backend)
   - 6.5 [UI Layer — Giao diện Streamlit](#65-ui-layer--giao-diện-streamlit)
7. [Hỗ trợ tiếng Việt](#7-hỗ-trợ-tiếng-việt)
8. [Công nghệ sử dụng](#8-công-nghệ-sử-dụng)
9. [Hướng dẫn cài đặt và sử dụng](#9-hướng-dẫn-cài-đặt-và-sử-dụng)
10. [Kết luận và hướng phát triển](#10-kết-luận-và-hướng-phát-triển)

---

## 1. Giới thiệu tổng quan

### 1.1 Bối cảnh và động lực

Trong bối cảnh các mô hình ngôn ngữ lớn (Large Language Models — LLM) ngày càng phổ biến, một trong những hạn chế lớn nhất của chúng là **hiện tượng "ảo giác" (hallucination)** — tức mô hình tạo ra thông tin sai lệch nhưng trông có vẻ hợp lý. Nguyên nhân chính là do LLM chỉ dựa trên kiến thức được học trong quá trình huấn luyện, không có khả năng truy cập thông tin mới hoặc dữ liệu chuyên ngành cụ thể.

**Retrieval-Augmented Generation (RAG)** ra đời để giải quyết vấn đề này bằng cách kết hợp khả năng sinh văn bản của LLM với cơ chế truy xuất thông tin từ nguồn dữ liệu bên ngoài. Tuy nhiên, các hệ thống RAG truyền thống vẫn còn nhiều hạn chế:

- **Chỉ xử lý văn bản thuần túy:** không thể hiểu hình ảnh, bảng biểu, công thức toán học trong tài liệu.
- **Truy xuất dựa trên vector đơn thuần:** chỉ tìm kiếm tương tự ngữ nghĩa, thiếu khả năng suy luận trên mối quan hệ giữa các khái niệm.
- **Thiếu hỗ trợ đa ngôn ngữ:** đặc biệt là tiếng Việt — một ngôn ngữ có cấu trúc phức tạp.

### 1.2 Mục tiêu của nhóm

Nhóm đặt ra mục tiêu xây dựng một hệ thống RAG toàn diện, giải quyết đồng thời các hạn chế trên:

1. **Đa phương thức (Multimodal):** Xử lý được văn bản, hình ảnh, bảng biểu, và công thức toán học trong cùng một pipeline.
2. **Đồ thị tri thức (Knowledge Graph):** Sử dụng đồ thị tri thức để nắm bắt mối quan hệ giữa các thực thể, cho phép suy luận sâu hơn.
3. **Hỗ trợ tiếng Việt:** Toàn bộ prompt, giao diện người dùng, và kết quả trả lời đều bằng tiếng Việt.
4. **Kiến trúc production-ready:** API RESTful, giao diện chat thân thiện, xử lý lỗi hoàn chỉnh.

### 1.3 Tổng quan giải pháp

Nhóm xây dựng hệ thống **RAG-Anything-UIT** bằng cách kết hợp ba công nghệ nền tảng:

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Phân tích tài liệu | MinerU | Trích xuất nội dung đa phương thức từ PDF/tài liệu |
| RAG Engine | LightRAG + RAG-Anything | Xây dựng đồ thị tri thức và truy vấn thông minh |
| LLM Provider | OpenRouter (Qwen models) | Sinh câu trả lời và phân tích nội dung |
| Vector Database | NanoVectorDB / Milvus Lite | Lưu trữ và tìm kiếm vector embedding |
| Backend API | FastAPI | Cung cấp REST API |
| Giao diện | Streamlit | Chat UI tiếng Việt |

---

## 2. Cơ sở lý thuyết

### 2.1 Retrieval-Augmented Generation (RAG)

#### 2.1.1 Định nghĩa

RAG (Retrieval-Augmented Generation) là một kiến trúc kết hợp hai thành phần chính:

1. **Retriever (Bộ truy xuất):** Tìm kiếm các đoạn văn bản liên quan từ kho dữ liệu dựa trên câu hỏi của người dùng.
2. **Generator (Bộ sinh):** Sử dụng LLM để sinh câu trả lời dựa trên ngữ cảnh được truy xuất.

#### 2.1.2 Quy trình hoạt động của RAG truyền thống

```
Câu hỏi người dùng
       │
       ▼
┌──────────────┐
│  Embedding   │  ← Chuyển câu hỏi thành vector
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Vector Search│  ← Tìm các chunk tương tự nhất
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Top-K      │  ← Chọn K chunk có điểm cao nhất
│   Chunks     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LLM Prompt  │  ← Ghép chunk vào prompt + câu hỏi
│  Generation  │
└──────┬───────┘
       │
       ▼
   Câu trả lời
```

#### 2.1.3 Hạn chế của RAG truyền thống

| Hạn chế | Mô tả |
|---|---|
| **Mất ngữ cảnh rộng** | Chunking chia nhỏ tài liệu, mất liên kết giữa các phần xa nhau |
| **Không nắm bắt quan hệ** | Vector search chỉ đo độ tương tự, không hiểu quan hệ nhân-quả, phân cấp |
| **Nhạy cảm với cách diễn đạt** | Câu hỏi phải dùng từ tương tự tài liệu mới tìm được |
| **Chỉ xử lý text** | Bỏ qua hình ảnh, bảng biểu, biểu đồ — nguồn thông tin quan trọng |

### 2.2 LightRAG — RAG dựa trên Đồ thị Tri thức

#### 2.2.1 Ý tưởng cốt lõi

LightRAG giải quyết hạn chế của RAG truyền thống bằng cách **xây dựng đồ thị tri thức (Knowledge Graph)** từ tài liệu. Thay vì chỉ lưu trữ các chunk văn bản dưới dạng vector, LightRAG trích xuất **thực thể (entities)** và **mối quan hệ (relationships)** giữa chúng, tạo thành một đồ thị có cấu trúc.

#### 2.2.2 Pipeline nhập liệu của LightRAG

LightRAG xử lý tài liệu qua 5 bước chính:

**Bước 1 — Enqueue (Xếp hàng)**
- Tài liệu được kiểm tra trùng lặp bằng hash MD5 của nội dung.
- Mỗi tài liệu được lưu vào `doc_status` storage với trạng thái `PENDING`.
- Thông tin metadata: đường dẫn file, kích thước, timestamp.

**Bước 2 — Chunking (Chia nhỏ)**
- Tài liệu được chia thành các chunk dựa trên số lượng token.
- Tham số mặc định: `chunk_token_size = 1200`, `chunk_overlap_token_size = 100`.
- Tokenizer mặc định: TikToken (model GPT-4o-mini).
- Mỗi chunk được gán `chunk_order_index` để giữ thứ tự ban đầu.
- Hỗ trợ chia theo ký tự (ví dụ: `\n\n`) trước khi chia theo token.

```
Tài liệu gốc (10,000 tokens)
       │
       ▼
┌─────────────────────────────────────────────────┐
│ Chunk 1      │ Chunk 2      │ ... │ Chunk N     │
│ (1200 tokens)│ (1200 tokens)│     │ (≤1200)     │
│              │              │     │             │
│    ←100→     │    ←100→     │     │             │
│   overlap    │   overlap    │     │             │
└─────────────────────────────────────────────────┘
```

**Bước 3 — Extract (Trích xuất thực thể và quan hệ)**

Đây là bước quan trọng nhất. LLM được sử dụng để trích xuất thực thể và quan hệ từ mỗi chunk.

*Định dạng trích xuất thực thể:*
```
entity<|#|>{tên_thực_thể}<|#|>{loại}<|#|>{mô_tả}
```

*Định dạng trích xuất quan hệ:*
```
relation<|#|>{nguồn}<|#|>{đích}<|#|>{từ_khóa}<|#|>{mô_tả}<|#|>{trọng_số}
```

*Ví dụ với một đoạn văn bản về quy chế đào tạo:*

```
Đầu vào (chunk):
"Nghiên cứu sinh phải hoàn thành tối thiểu 90 tín chỉ trong thời gian
3 năm. Hội đồng khoa học xét duyệt đề cương nghiên cứu hàng năm."

Đầu ra (entities):
entity<|#|>Nghiên cứu sinh<|#|>person<|#|>Đối tượng đào tạo bậc tiến sĩ
entity<|#|>Tín chỉ<|#|>concept<|#|>Đơn vị đo lường khối lượng học tập
entity<|#|>Hội đồng khoa học<|#|>organization<|#|>Cơ quan xét duyệt học thuật
entity<|#|>Đề cương nghiên cứu<|#|>document<|#|>Bản mô tả kế hoạch nghiên cứu

Đầu ra (relations):
relation<|#|>Nghiên cứu sinh<|#|>Tín chỉ<|#|>yêu cầu<|#|>NCS phải hoàn thành tối thiểu 90 tín chỉ<|#|>9
relation<|#|>Hội đồng khoa học<|#|>Đề cương nghiên cứu<|#|>xét duyệt<|#|>HĐ xét duyệt đề cương hàng năm<|#|>8
```

*Cơ chế Gleaning (Rà soát lại):*
- Sau lần trích xuất đầu tiên, LLM được yêu cầu rà soát lại để tìm các thực thể/quan hệ bị bỏ sót.
- Số lần gleaning tối đa: `entity_extract_max_gleaning = 10`.
- So sánh độ dài mô tả, giữ lại phiên bản tốt hơn.

**Bước 4 — Merge (Hợp nhất)**

Khi cùng một thực thể xuất hiện trong nhiều chunk, LightRAG thực hiện hợp nhất:

- **Hợp nhất thực thể (`_merge_nodes_then_upsert`):**
  1. Lấy dữ liệu node hiện có trong đồ thị.
  2. Hợp nhất source IDs mới với source IDs cũ.
  3. Loại bỏ mô tả trùng lặp (deduplicate).
  4. Nếu số mô tả vượt ngưỡng, sử dụng LLM tóm tắt theo phương pháp **Map-Reduce**.
  5. Cập nhật đồ thị và vector database.

- **Hợp nhất quan hệ (`_merge_edges_then_upsert`):**
  1. Tương tự thực thể, nhưng thêm: hợp nhất từ khóa, cộng dồn trọng số.
  2. Đảm bảo node đầu mút tồn tại (tạo mới nếu thiếu).

- **Phương pháp Map-Reduce cho tóm tắt mô tả:**
  ```
  Nhiều mô tả từ nhiều chunk
         │
         ▼
  ┌──────────────────┐
  │  Chia thành nhóm │  (nếu tổng token > ngưỡng)
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │  LLM tóm tắt     │  (mỗi nhóm → 1 bản tóm tắt)
  │  từng nhóm        │
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │  Đệ quy merge   │  (nếu vẫn còn nhiều, lặp lại)
  └──────┬───────────┘
         │
         ▼
     Mô tả hợp nhất cuối cùng
  ```

**Bước 5 — Store (Lưu trữ)**

Dữ liệu được lưu vào nhiều lớp storage:

| Loại Storage | Mục đích | Backend mặc định |
|---|---|---|
| **Vector DB** (entities, relationships, chunks) | Tìm kiếm ngữ nghĩa | NanoVectorDB |
| **Graph DB** (đồ thị tri thức) | Lưu cấu trúc đồ thị | NetworkX (GraphML) |
| **KV Store** (metadata) | Lưu thông tin chi tiết | JSON files |
| **Doc Status** (trạng thái tài liệu) | Theo dõi quá trình xử lý | JSON files |
| **LLM Cache** (cache phản hồi) | Tránh gọi LLM trùng lặp | JSON files |

#### 2.2.3 Pipeline truy vấn của LightRAG

LightRAG cung cấp **6 chế độ truy vấn** khác nhau:

| Chế độ | Chiến lược | Khi nào sử dụng |
|---|---|---|
| **naive** | Tìm kiếm vector trên chunk thuần | Nhanh, không dùng đồ thị tri thức |
| **local** | Tìm thực thể → truy vết quan hệ → lấy chunk | Câu hỏi chi tiết, cụ thể |
| **global** | Tìm quan hệ → truy vết thực thể → lấy chunk | Câu hỏi tổng quan, khái quát |
| **hybrid** | Kết hợp local + global | Câu hỏi cân bằng (mặc định) |
| **mix** | Đồ thị tri thức + vector thuần | Tận dụng cả hai nguồn |
| **bypass** | Gửi thẳng cho LLM, không truy xuất | Test/so sánh |

*Chi tiết quy trình truy vấn:*

```
Câu hỏi người dùng
       │
       ▼
┌─────────────────────┐
│ Trích xuất từ khóa  │  ← LLM phân tích: high-level + low-level keywords
└──────┬──────────────┘
       │
       ├─── Local Mode ──────────────────────────┐
       │    Vector search trên entities_vdb       │
       │    → Tìm node trong đồ thị              │
       │    → Duyệt cạnh (edge) liên quan        │
       │    → Lấy chunk gốc qua source_ids       │
       │                                          │
       ├─── Global Mode ─────────────────────────┤
       │    Vector search trên relationships_vdb  │
       │    → Tìm cạnh trong đồ thị              │
       │    → Truy ngược node đầu mút            │
       │    → Lấy chunk gốc qua source_ids       │
       │                                          │
       ├─── Hybrid Mode ─────────────────────────┤
       │    Kết hợp kết quả Local + Global        │
       │    Token-aware merging                   │
       │                                          │
       ├─── Mix Mode ────────────────────────────┤
       │    Hybrid + Naive vector search          │
       │    Merge tất cả kết quả                  │
       │                                          │
       └──────────────┬──────────────────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │ Reranking        │  ← Cohere Rerank v3.5
            │ (tùy chọn)       │
            └──────┬───────────┘
                   │
                   ▼
            ┌──────────────────┐
            │ Token Truncation │  ← Cắt ngữ cảnh theo giới hạn token
            │ - entities: 256  │
            │ - relations: 256 │
            │ - total: 4000    │
            └──────┬───────────┘
                   │
                   ▼
            ┌──────────────────┐
            │ LLM Generation   │  ← Sinh câu trả lời từ ngữ cảnh
            └──────┬───────────┘
                   │
                   ▼
              Câu trả lời
```

*Phương pháp chọn chunk (`kg_chunk_pick_method`):*

- **WEIGHT (mặc định):** Chọn chunk dựa trên tần suất xuất hiện trong kết quả — chunk được nhiều entity/relation tham chiếu sẽ có trọng số cao hơn.
- **VECTOR:** Chọn chunk dựa trên cosine similarity giữa embedding của chunk và embedding của câu hỏi.

#### 2.2.4 Xử lý đồng thời và caching

LightRAG hỗ trợ xử lý song song với các tham số:

| Tham số | Giá trị mặc định | Mô tả |
|---|---|---|
| `max_parallel_insert` | 16 | Số tài liệu xử lý song song |
| `llm_model_max_async` | 4 | Số lệnh gọi LLM đồng thời |
| `embedding_func_max_async` | 8 | Số lệnh gọi embedding đồng thời |
| `embedding_batch_num` | 10 | Số text embedding trong 1 batch |

Hệ thống caching LLM response:
- Mỗi phản hồi LLM được cache với key là hash của prompt.
- Cache phân loại theo `cache_type`: `extract`, `query`, `keywords`, `summary`.
- Tránh gọi LLM trùng lặp khi xử lý cùng nội dung.

### 2.3 RAG-Anything — Mở rộng đa phương thức

#### 2.3.1 Vai trò của RAG-Anything

RAG-Anything là một lớp mở rộng phía trên LightRAG, bổ sung khả năng **xử lý nội dung đa phương thức**. Trong khi LightRAG chỉ xử lý văn bản, RAG-Anything cho phép hệ thống hiểu và tích hợp:

- **Hình ảnh:** Biểu đồ, sơ đồ, ảnh minh họa, ảnh chụp
- **Bảng biểu:** Bảng dữ liệu, bảng so sánh, bảng thống kê
- **Công thức toán học:** Phương trình, biểu thức LaTeX
- **Nội dung generic:** Bất kỳ loại nội dung đặc biệt nào khác

#### 2.3.2 Cách RAG-Anything xử lý nội dung đa phương thức

```
Tài liệu PDF
       │
       ▼
┌─────────────────────┐
│     MinerU Parser    │  ← Trích xuất cấu trúc tài liệu
└──────┬──────────────┘
       │
       ▼
  content_list.json
  [
    {"type": "text", "content": "..."},
    {"type": "image", "img_path": "...", "captions": "..."},
    {"type": "table", "table_body": "...", "table_caption": "..."},
    {"type": "equation", "equation_text": "...", "equation_format": "latex"},
    ...
  ]
       │
       ▼
┌──────────────────────────────────────────────────────┐
│              RAG-Anything Processing                  │
│                                                       │
│  Text items:                                          │
│    → Chunking trực tiếp → Embedding                   │
│                                                       │
│  Image items:                                         │
│    → VLM phân tích (Qwen2.5-VL-72B) → Mô tả chi tiết│
│    → Chuyển mô tả thành text → Chunking → Embedding  │
│                                                       │
│  Table items:                                         │
│    → LLM phân tích cấu trúc → Tóm tắt ý nghĩa      │
│    → Chuyển thành text → Chunking → Embedding         │
│                                                       │
│  Equation items:                                      │
│    → LLM giải thích toán học → Mô tả ý nghĩa        │
│    → Chuyển thành text → Chunking → Embedding         │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
              LightRAG KG Pipeline
              (Trích xuất entity/relation → Đồ thị tri thức)
```

#### 2.3.3 Context Window — Ngữ cảnh xung quanh

Một tính năng quan trọng của RAG-Anything là **context window** — khả năng cung cấp ngữ cảnh xung quanh cho mỗi phần tử nội dung:

- `context_window = 1`: Lấy 1 trang trước và 1 trang sau phần tử hiện tại.
- `context_mode = "page"`: Đơn vị ngữ cảnh theo trang tài liệu.
- `max_context_tokens = 2000`: Giới hạn token cho ngữ cảnh.

Điều này giúp VLM hiểu hình ảnh/bảng trong bối cảnh — ví dụ: một biểu đồ nằm trong chương nào, minh họa cho nội dung gì.

---

## 3. Thiết kế hệ thống

### 3.1 Kiến trúc tổng thể

Nhóm thiết kế hệ thống theo kiến trúc **layered (phân tầng)** kết hợp với **dependency injection**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        UI Layer                                  │
│                    (Streamlit Chat)                               │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ - Giao diện chat tiếng Việt                              │     │
│  │ - Upload tài liệu (PDF, DOCX, PPTX, XLSX, images)       │     │
│  │ - Cấu hình truy vấn (mode, top_k, response_type)        │     │
│  │ - Hiển thị thông tin hệ thống                            │     │
│  └────────────────────────┬────────────────────────────────┘     │
│                           │ HTTP (REST)                          │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                        API Layer                                 │
│                     (FastAPI Backend)                             │
│  ┌────────────────────────┴────────────────────────────────┐     │
│  │ Endpoints:                                                │     │
│  │   POST /chat         — Truy vấn RAG                      │     │
│  │   POST /ingest       — Nhập tài liệu từ đường dẫn       │     │
│  │   POST /ingest/upload — Nhập tài liệu upload            │     │
│  │   GET  /health       — Kiểm tra sức khỏe hệ thống       │     │
│  │   GET  /system/info  — Thông tin cấu hình                │     │
│  └────────────────────────┬────────────────────────────────┘     │
│                           │ Dependency Injection                 │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                     Service Layer                                │
│               (Business Logic + Orchestration)                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ ServiceContainer (Lifecycle Manager)                      │     │
│  │   ├── RAGService.create() → RAGAnything instance          │     │
│  │   ├── IngestionService → Document processing              │     │
│  │   └── startup() / shutdown() lifecycle                    │     │
│  └────────────────────────┬────────────────────────────────┘     │
│                           │                                      │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                    Adapter Layer                                  │
│              (External Service Integration)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐      │
│  │ LLMAdapter   │  │EmbeddingAdap.│  │ Reranker (Cohere) │      │
│  │ - chat()     │  │ - embed_text │  │ - rerank_func()   │      │
│  │ - chat_with_ │  │ - embed_texts│  │                   │      │
│  │   image()    │  │              │  │                   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬────────────┘      │
│         │                 │                  │                    │
└─────────┼─────────────────┼──────────────────┼──────────────────┘
          │                 │                  │
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   External Services                              │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              OpenRouter API                                │   │
│  │  ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐  │   │
│  │  │ Qwen3-30B-A3B   │ │ Qwen2.5-VL   │ │ Qwen3-Embed   │  │   │
│  │  │ (Text LLM)      │ │ -72B (Vision)│ │ -8B           │  │   │
│  │  └─────────────────┘ └──────────────┘ └───────────────┘  │   │
│  │  ┌─────────────────┐                                      │   │
│  │  │ Cohere Rerank   │                                      │   │
│  │  │ v3.5            │                                      │   │
│  │  └─────────────────┘                                      │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Storage Layer                                  │
│  ┌──────────────┐ ┌───────────────┐ ┌────────────────────────┐  │
│  │ NanoVectorDB │ │ NetworkX      │ │ JSON KV Store          │  │
│  │ (Vector DB)  │ │ (Graph DB)    │ │ (Metadata)             │  │
│  │              │ │               │ │                        │  │
│  │ - entities   │ │ - GraphML     │ │ - text_chunks          │  │
│  │ - relations  │ │ - nodes       │ │ - full_docs            │  │
│  │ - chunks     │ │ - edges       │ │ - entity/relation data │  │
│  └──────────────┘ └───────────────┘ └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Cấu trúc thư mục dự án

```
RAG-Anything-UIT/
├── rag_app/                       # Package chính của nhóm
│   ├── __init__.py
│   ├── core/                      # Hạ tầng cốt lõi
│   │   ├── config.py              # Cấu hình (Pydantic Settings)
│   │   ├── exceptions.py          # Hệ thống ngoại lệ
│   │   └── logging.py             # Thiết lập logging
│   ├── adapters/                  # Tích hợp dịch vụ bên ngoài
│   │   ├── base.py                # HTTP client cơ sở (retry, backoff)
│   │   ├── llm.py                 # LLM Adapter (text + vision)
│   │   └── embeddings.py          # Embedding Adapter
│   ├── store/                     # Lưu trữ vector
│   │   └── milvus.py              # Milvus Lite wrapper
│   ├── services/                  # Logic nghiệp vụ
│   │   ├── rag.py                 # Khởi tạo RAGAnything
│   │   ├── ingest.py              # Dịch vụ nhập tài liệu
│   │   └── container.py           # Dependency Injection container
│   ├── api/                       # FastAPI application
│   │   ├── app.py                 # App factory + CORS
│   │   ├── routes.py              # Định tuyến API
│   │   └── deps.py                # FastAPI dependencies
│   └── prompts_vi.py              # Prompt tiếng Việt
│
├── run_api.py                     # Khởi động FastAPI server
├── run_ui.py                      # Khởi động Streamlit UI
├── ingest_pdf.py                  # CLI: nhập tài liệu PDF
├── query_system.py                # CLI: truy vấn hệ thống
├── reingest.py                    # CLI: nhập lại từ content đã parse
│
├── rag_workdir/                   # Thư mục dữ liệu runtime
│   ├── kv_store_*.json            # KV stores (entities, relations, chunks)
│   ├── vdb_*.json                 # Vector database records
│   └── graph_chunk_entity_relation.graphml  # Đồ thị tri thức
│
├── output/                        # Kết quả MinerU extraction
├── sample/docs/                   # Tài liệu mẫu
├── pyproject.toml                 # Metadata & dependencies
├── requirements.txt               # Dependencies (pip format)
├── .env.example                   # Template cấu hình
└── .env                           # Cấu hình thực tế (chứa API key)
```

### 3.3 Các thành phần chính

#### 3.3.1 Core Module (`rag_app/core/`)

**config.py — Quản lý cấu hình:**

Nhóm sử dụng **Pydantic Settings** để quản lý toàn bộ cấu hình hệ thống. Tất cả tham số được đọc từ file `.env` và validate tự động:

```python
class Settings(BaseSettings):
    # API — bắt buộc, không có giá trị mặc định
    openrouter_api_key: str

    # Mô hình LLM
    llm_text_model: str = "qwen/qwen3-30b-a3b"        # Text generation
    llm_vlm_model: str = "qwen/qwen2.5-vl-72b-instruct"  # Vision
    embed_model: str = "qwen/qwen3-embedding-8b"       # Embedding
    embed_dim: int = 4096                               # Chiều vector

    # Reranker
    reranker_enabled: bool = True
    reranker_model: str = "cohere/rerank-v3.5"

    # Ngôn ngữ
    summary_language: str = "Vietnamese"
    query_user_prompt: str = "Always respond in Vietnamese..."

    # RAGAnything processing
    enable_image_processing: bool = True
    enable_table_processing: bool = True
    enable_equation_processing: bool = True
    rag_parser: str = "mineru"
    context_window: int = 1           # Số trang ngữ cảnh
    context_mode: str = "page"        # Đơn vị ngữ cảnh
    max_context_tokens: int = 2000    # Giới hạn token ngữ cảnh
```

**exceptions.py — Hệ thống ngoại lệ:**

Nhóm thiết kế hệ thống ngoại lệ phân cấp để xử lý lỗi chính xác:

```
RAGAppError (base)
├── ConfigurationError     # Lỗi cấu hình (thiếu API key, giá trị không hợp lệ)
├── AdapterError           # Lỗi giao tiếp API
│   └── RetryExhaustedError  # Đã hết số lần thử lại
├── VectorStoreError       # Lỗi Milvus
└── IngestionError         # Lỗi nhập tài liệu
```

#### 3.3.2 Adapter Layer (`rag_app/adapters/`)

**BaseOpenRouterClient — HTTP Client cơ sở:**

Nhóm thiết kế một lớp HTTP client dùng chung với các tính năng:

- **Session reuse:** Tái sử dụng kết nối HTTP thông qua `aiohttp.ClientSession`.
- **Exponential backoff retry:** Tự động thử lại với thời gian chờ tăng dần (2^attempt giây, tối đa 30 giây).
- **Retryable status codes:** Tự động retry cho các mã HTTP: 429 (rate limit), 500, 502, 503, 504.
- **Bearer token authentication:** Xác thực qua OpenRouter API key.

```python
async def _post_with_retry(self, url, payload):
    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status in {429, 500, 502, 503, 504}:
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
                return await resp.json()
        except (ClientError, TimeoutError):
            await asyncio.sleep(min(2 ** attempt, 30))
    raise RetryExhaustedError(...)
```

**LLMAdapter — Giao tiếp LLM:**

Cung cấp hai phương thức chính:

- `chat()`: Gọi LLM text-only (Qwen3-30B-A3B) với system prompt, history messages, temperature control.
- `chat_with_image()`: Gọi Vision LLM (Qwen2.5-VL-72B) với hình ảnh base64-encoded.

Đặc biệt, LLMAdapter tự động loại bỏ **thinking tokens** (`<think>...</think>`) từ phản hồi của các thinking models:

```python
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def _extract_content(data):
    content = data["choices"][0]["message"].get("content")
    return _THINK_RE.sub("", content).strip()
```

**EmbeddingAdapter — Tạo vector embedding:**

- `embed_text()`: Embedding cho một đoạn text.
- `embed_texts()`: Embedding batch cho nhiều đoạn text cùng lúc.

#### 3.3.3 Service Layer (`rag_app/services/`)

**ServiceContainer — Dependency Injection:**

Đây là thành phần trung tâm quản lý vòng đời của tất cả service:

```python
class ServiceContainer:
    async def startup(self):
        self.llm = LLMAdapter(settings)      # Khởi tạo LLM client
        self.emb = EmbeddingAdapter(settings) # Khởi tạo Embedding client
        self.rag = await RAGService.create(settings, self.llm, self.emb)  # Khởi tạo RAG

    async def shutdown(self):
        await asyncio.gather(self.llm.close(), self.emb.close())  # Đóng sessions
```

**RAGService — Khởi tạo RAGAnything:**

RAGService đóng vai trò **factory** để tạo instance RAGAnything với đầy đủ cấu hình:

1. Đăng ký prompt tiếng Việt nếu `SUMMARY_LANGUAGE = "Vietnamese"`.
2. Tạo wrapper functions cho LLM và embedding.
3. Cấu hình reranker (Cohere Rerank v3.5) nếu được bật.
4. Tạo `RAGAnythingConfig` với các tùy chọn xử lý (image, table, equation, parser).
5. Khởi tạo `RAGAnything` instance và đảm bảo LightRAG đã sẵn sàng.

**IngestionService — Nhập tài liệu:**

Cung cấp interface đơn giản để nhập tài liệu:

```python
async def ingest_document(self, path, *, start_page=None, end_page=None):
    await self._rag.process_document_complete(str(path), device="cuda", **kwargs)
```

---

## 4. Pipeline nhập liệu tài liệu (Ingestion)

### 4.1 Giai đoạn 1: Phân tích tài liệu (MinerU)

**MinerU** là bộ phân tích tài liệu (document parser) được sử dụng để trích xuất nội dung có cấu trúc từ file PDF. MinerU hỗ trợ:

- **Phát hiện layout tự động:** Nhận diện vùng text, hình ảnh, bảng, công thức.
- **OCR tích hợp:** Nhận dạng văn bản trong hình ảnh.
- **GPU acceleration:** Sử dụng CUDA để tăng tốc xử lý.

*Quy trình:*

```
PDF Input
    │
    ▼
┌──────────────────┐
│  Layout Detection│  ← Phát hiện vùng nội dung trên mỗi trang
│  (GPU/CUDA)      │
└──────┬───────────┘
    │
    ▼
┌──────────────────┐
│  Content Extract │
│  ├── Text blocks │  ← Trích xuất đoạn văn bản
│  ├── Images      │  ← Cắt và lưu hình ảnh
│  ├── Tables      │  ← Nhận dạng cấu trúc bảng
│  └── Equations   │  ← Trích xuất công thức LaTeX
└──────┬───────────┘
    │
    ▼
┌──────────────────┐
│  Output Files    │
│  ├── content.md           ← Markdown tổng hợp
│  ├── content_list.json    ← Danh sách nội dung có cấu trúc
│  ├── model.json           ← Metadata mô hình
│  ├── layout.pdf           ← PDF với annotation layout
│  └── images/              ← Thư mục hình ảnh trích xuất
└──────────────────┘
```

*Cấu hình MinerU trong hệ thống:*

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `rag_parser` | `"mineru"` | Sử dụng MinerU parser |
| `parse_method` | `"auto"` | Tự động chọn phương pháp |
| `mineru_device` | `"cuda"` | GPU acceleration |

### 4.2 Giai đoạn 2: Phân tích nội dung đa phương thức

Sau khi MinerU trích xuất, RAG-Anything xử lý từng loại nội dung:

#### 4.2.1 Xử lý hình ảnh

```
Hình ảnh (PNG/JPG)
       │
       ▼
┌─────────────────────────┐
│ VLM (Qwen2.5-VL-72B)   │
│                          │
│ System prompt:           │
│ "Bạn là chuyên gia      │
│  phân tích hình ảnh..."  │
│                          │
│ Input:                   │
│ - Hình ảnh (base64)      │
│ - Chú thích (captions)   │
│ - Ghi chú (footnotes)    │
│ - Ngữ cảnh xung quanh   │
│                          │
│ Output (JSON):           │
│ {                        │
│   "detailed_description":│
│     "Mô tả chi tiết...",│
│   "entity_info": {       │
│     "entity_name": "...",│
│     "entity_type": "img",│
│     "summary": "..."     │
│   }                      │
│ }                        │
└──────────┬──────────────┘
           │
           ▼
    Enhanced Caption (text)
    → Đưa vào LightRAG pipeline
```

VLM phân tích hình ảnh theo các tiêu chí:
- Bố cục tổng thể và cách trình bày
- Nhận diện đối tượng, con người, văn bản
- Mối quan hệ giữa các yếu tố
- Màu sắc, ánh sáng, phong cách
- Chi tiết kỹ thuật (nếu là biểu đồ/sơ đồ)

#### 4.2.2 Xử lý bảng biểu

```
Bảng dữ liệu
       │
       ▼
┌─────────────────────────┐
│ LLM (Qwen3-30B-A3B)     │
│                          │
│ System prompt:           │
│ "Bạn là chuyên gia      │
│  phân tích dữ liệu..."  │
│                          │
│ Input:                   │
│ - Nội dung bảng (HTML)   │
│ - Tiêu đề bảng           │
│ - Hình ảnh bảng (nếu có) │
│ - Ghi chú                │
│                          │
│ Output: Mô tả ngữ nghĩa │
│ của cấu trúc bảng,      │
│ xu hướng dữ liệu,       │
│ phát hiện quan trọng     │
└──────────┬──────────────┘
           │
           ▼
    Enhanced Caption (text)
```

#### 4.2.3 Xử lý công thức toán học

```
Công thức LaTeX
       │
       ▼
┌─────────────────────────┐
│ LLM (Qwen3-30B-A3B)     │
│                          │
│ System prompt:           │
│ "Bạn là chuyên gia      │
│  toán học..."            │
│                          │
│ Input:                   │
│ - Biểu thức LaTeX        │
│ - Định dạng              │
│ - Ngữ cảnh (nếu có)     │
│                          │
│ Output: Giải thích       │
│ ý nghĩa toán học,       │
│ biến số, ứng dụng       │
└──────────┬──────────────┘
           │
           ▼
    Enhanced Caption (text)
```

### 4.3 Giai đoạn 3: Xây dựng đồ thị tri thức

Sau khi tất cả nội dung đa phương thức đã được chuyển thành text (enhanced captions), toàn bộ được đưa vào pipeline LightRAG:

```
Enhanced Captions + Original Text
              │
              ▼
┌─────────────────────────┐
│ Chunking (1200 tokens)  │
└──────────┬──────────────┘
              │
              ▼
┌─────────────────────────┐
│ Entity/Relation Extract │  ← LLM trích xuất
│ + Gleaning              │
└──────────┬──────────────┘
              │
              ▼
┌─────────────────────────┐
│ Merge & Deduplicate     │  ← Hợp nhất, loại trùng
└──────────┬──────────────┘
              │
              ▼
┌─────────────────────────┐
│ Multi-Store Persistence │
│ ├── Vector DB (3 stores)│
│ ├── Graph DB (GraphML)  │
│ └── KV Store (JSON)     │
└─────────────────────────┘
```

*Kết quả lưu trữ trong `rag_workdir/`:*

| File | Nội dung |
|---|---|
| `graph_chunk_entity_relation.graphml` | Đồ thị tri thức (nodes + edges) |
| `vdb_entities.json` | Vector embeddings của thực thể |
| `vdb_relationships.json` | Vector embeddings của quan hệ |
| `vdb_chunks.json` | Vector embeddings của chunks |
| `kv_store_text_chunks.json` | Nội dung text của chunks |
| `kv_store_full_docs.json` | Nội dung đầy đủ tài liệu |
| `kv_store_llm_response_cache.json` | Cache phản hồi LLM |

---

## 5. Pipeline truy vấn (Query)

### 5.1 Trích xuất từ khóa

Khi nhận câu hỏi từ người dùng, bước đầu tiên là trích xuất từ khóa:

```
Câu hỏi: "Nghiên cứu sinh cần hoàn thành bao nhiêu tín chỉ?"
                    │
                    ▼
            ┌──────────────┐
            │     LLM       │
            └──────┬───────┘
                   │
    ┌──────────────┼──────────────────┐
    │              │                   │
    ▼              ▼                   ▼
High-level     Low-level           Kết hợp
Keywords       Keywords
"đào tạo       "nghiên cứu sinh"   → Dùng cho
 sau đại học"   "tín chỉ"            truy vấn KG
"chương trình   "yêu cầu            + vector
 tiến sĩ"       tốt nghiệp"         search
```

- **High-level keywords:** Từ khóa khái quát, dùng cho Global mode (tìm quan hệ).
- **Low-level keywords:** Từ khóa cụ thể, dùng cho Local mode (tìm thực thể).

### 5.2 Các chế độ truy vấn

#### 5.2.1 Local Mode

```
Low-level keywords: ["nghiên cứu sinh", "tín chỉ"]
         │
         ▼
┌─────────────────────┐
│ Vector Search trên   │
│ entities_vdb         │  → Tìm entity "Nghiên cứu sinh", "Tín chỉ"
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Duyệt edges trong   │
│ Knowledge Graph      │  → Tìm quan hệ "NCS → Tín chỉ: yêu cầu 90"
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Truy ngược chunks   │
│ qua source_ids      │  → Lấy đoạn văn gốc chứa thông tin
└──────┬──────────────┘
       │
       ▼
  Ngữ cảnh cho LLM
```

#### 5.2.2 Global Mode

```
High-level keywords: ["đào tạo sau đại học", "chương trình tiến sĩ"]
         │
         ▼
┌─────────────────────┐
│ Vector Search trên   │
│ relationships_vdb    │  → Tìm quan hệ liên quan đến đào tạo
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Truy ngược entities │
│ qua node đầu mút    │  → Lấy thực thể liên quan
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Truy ngược chunks   │
│ qua source_ids      │
└──────┬──────────────┘
       │
       ▼
  Ngữ cảnh cho LLM
```

#### 5.2.3 Hybrid Mode (Mặc định)

Kết hợp kết quả từ Local và Global, sau đó merge theo token budget:

```
Local results + Global results
         │
         ▼
┌─────────────────────┐
│ Token-aware Merge    │
│ - entities: max 256  │
│ - relations: max 256 │
│ - total: max 4000    │
└──────┬──────────────┘
         │
         ▼
   Merged context cho LLM
```

#### 5.2.4 Mix Mode

Kết hợp Hybrid với Naive (vector search thuần trên chunks):

```
Hybrid results + Naive vector search results
         │
         ▼
   Merged + Deduplicated context
```

### 5.3 Reranking

Nhóm tích hợp **Cohere Rerank v3.5** qua OpenRouter API để sắp xếp lại kết quả truy xuất:

```
Danh sách chunks/entities/relations từ retrieval
         │
         ▼
┌─────────────────────────────┐
│ Cohere Rerank v3.5           │
│                              │
│ Input:                       │
│ - query: câu hỏi người dùng │
│ - documents: danh sách chunks│
│ - top_n: số kết quả cần giữ  │
│                              │
│ Output:                      │
│ - Danh sách đã sắp xếp lại  │
│   theo relevance score       │
└──────────┬──────────────────┘
           │
           ▼
   Chunks đã reranked → LLM
```

Reranker giúp:
- Loại bỏ kết quả không liên quan mà vector search trả về nhầm.
- Sắp xếp các kết quả theo thứ tự liên quan nhất.
- Cải thiện đáng kể chất lượng câu trả lời.

### 5.4 Sinh câu trả lời

```
┌───────────────────────────────────────────────┐
│ LLM Prompt Construction                       │
│                                                │
│ System: "Always respond in Vietnamese..."      │
│                                                │
│ Context:                                       │
│   Entities: [thực thể 1, thực thể 2, ...]     │
│   Relations: [quan hệ 1, quan hệ 2, ...]      │
│   Chunks: [đoạn văn 1, đoạn văn 2, ...]       │
│                                                │
│ Conversation History: (nếu có)                 │
│   User: câu hỏi trước                         │
│   Assistant: câu trả lời trước                 │
│                                                │
│ User: câu hỏi hiện tại                         │
│                                                │
│ User Prompt: "Always respond in Vietnamese..." │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
                  Qwen3-30B-A3B
                        │
                        ▼
               Câu trả lời tiếng Việt
```

---

## 6. Chi tiết triển khai

### 6.1 Adapter Layer — Giao tiếp với LLM

#### 6.1.1 BaseOpenRouterClient

Nhóm thiết kế lớp base client để tái sử dụng logic HTTP chung:

```python
class BaseOpenRouterClient:
    """Shared HTTP client with session reuse, retry, and exponential backoff."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._settings.timeout),
                headers={
                    "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def _post_with_retry(self, url, payload):
        for attempt in range(1, max_retries + 1):
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status in RETRYABLE_STATUS:  # {429, 500, 502, 503, 504}
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except (ClientError, TimeoutError):
                await asyncio.sleep(min(2 ** attempt, 30))
        raise RetryExhaustedError(attempts=max_retries, last_error=last_err)
```

**Đặc điểm thiết kế:**
- **Session reuse:** Không tạo connection mới cho mỗi request.
- **Exponential backoff:** 2s → 4s → 8s → ... → 30s (max).
- **Graceful degradation:** Retry tự động cho transient errors.

#### 6.1.2 LLMAdapter

Hai phương thức chính:

**`chat()` — Text-only LLM:**
- Model: `qwen/qwen3-30b-a3b`
- Temperature: 0.5 (cân bằng sáng tạo/nhất quán)
- Hỗ trợ system prompt, conversation history
- Tự động loại bỏ thinking tokens

**`chat_with_image()` — Vision LLM:**
- Model: `qwen/qwen2.5-vl-72b-instruct`
- Temperature: 0.2 (chính xác cao cho phân tích hình ảnh)
- Hỗ trợ input: file path hoặc base64 data
- Fallback: nếu không có hình ảnh → chuyển sang text-only chat

```python
async def chat_with_image(self, prompt, image_path=None, *, image_data=None, ...):
    if image_data:
        messages = self._build_messages(prompt, image_data, "image/png", system_prompt)
    elif image_path:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        messages = self._build_messages(prompt, b64, _detect_mime(image_path), ...)
    else:
        return await self.chat(prompt, ...)  # Fallback text-only
```

#### 6.1.3 EmbeddingAdapter

```python
class EmbeddingAdapter(BaseOpenRouterClient):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self._settings.embed_model, "input": texts}
        data = await self._post_with_retry(self._embed_url, payload)
        return [item["embedding"] for item in data["data"]]
```

- Model: `qwen/qwen3-embedding-8b`
- Dimension: 4096
- Hỗ trợ batch embedding

### 6.2 Storage Layer — Lưu trữ dữ liệu

#### 6.2.1 NanoVectorDB (Mặc định)

NanoVectorDB là vector database nhẹ, lưu trữ dưới dạng JSON files. Nhóm chọn NanoVectorDB làm backend mặc định vì:

- **Không cần cài đặt thêm:** Chạy trực tiếp trên file system.
- **Phù hợp prototype:** Nhanh chóng bắt đầu phát triển.
- **Dễ debug:** Dữ liệu lưu dạng JSON, có thể đọc trực tiếp.

Ba vector store riêng biệt:
- `vdb_entities.json`: Embedding của thực thể
- `vdb_relationships.json`: Embedding của quan hệ
- `vdb_chunks.json`: Embedding của chunks văn bản

#### 6.2.2 Milvus Lite (Tùy chọn)

Nhóm cũng hỗ trợ Milvus Lite cho production:

```python
class MilvusVectorStore:
    # Schema:
    # - id: VARCHAR (primary key)
    # - vector: FLOAT_VECTOR (dim=4096)
    # - content: VARCHAR (max 65535 chars)
    # - content_type: VARCHAR
    # - source: VARCHAR
    # - ts: INT64 (timestamp)

    async def search(self, query_vector, top_k=10, content_type=None):
        # COSINE similarity search với AUTOINDEX
        # Hỗ trợ filter theo content_type
```

Chuyển đổi giữa NanoVectorDB và Milvus bằng biến môi trường:
```
VECTOR_STORAGE=NanoVectorDBStorage    # Mặc định
VECTOR_STORAGE=MilvusVectorDBStorage  # Production
```

#### 6.2.3 NetworkX Graph Storage

Đồ thị tri thức được lưu dưới định dạng **GraphML** — chuẩn XML mở cho đồ thị:

```xml
<!-- graph_chunk_entity_relation.graphml -->
<graphml>
  <graph edgedefault="undirected">
    <node id="nghiên cứu sinh">
      <data key="entity_type">person</data>
      <data key="description">Đối tượng đào tạo bậc tiến sĩ...</data>
      <data key="source_ids">["chunk_001", "chunk_015"]</data>
    </node>
    <edge source="nghiên cứu sinh" target="tín chỉ">
      <data key="description">NCS phải hoàn thành tối thiểu 90 tín chỉ</data>
      <data key="keywords">yêu cầu, tốt nghiệp</data>
      <data key="weight">9.0</data>
    </edge>
  </graph>
</graphml>
```

### 6.3 Service Layer — Logic nghiệp vụ

#### 6.3.1 ServiceContainer — Quản lý vòng đời

```python
class ServiceContainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm: LLMAdapter | None = None
        self.emb: EmbeddingAdapter | None = None
        self.rag: RAGAnything | None = None

    async def startup(self):
        self.llm = LLMAdapter(self.settings)
        self.emb = EmbeddingAdapter(self.settings)
        self.rag = await RAGService.create(self.settings, self.llm, self.emb)

    async def shutdown(self):
        await asyncio.gather(self.llm.close(), self.emb.close())
```

ServiceContainer đảm bảo:
- **Khởi tạo đúng thứ tự:** LLM → Embedding → RAG.
- **Dọn dẹp tài nguyên:** Đóng HTTP sessions khi shutdown.
- **Thread safety:** Sử dụng async/await xuyên suốt.

#### 6.3.2 RAGService — Factory Pattern

```python
class RAGService:
    @staticmethod
    async def create(settings, llm, emb) -> RAGAnything:
        # 1. Đăng ký prompt tiếng Việt
        if settings.summary_language.lower() == "vietnamese":
            register_prompt_language("vi", PROMPTS_VI)
            set_prompt_language("vi")

        # 2. Wrap LLM function
        async def llm_func(prompt, **kwargs):
            return await llm.chat(prompt, system_prompt=kwargs.get("system_prompt"), ...)

        # 3. Wrap embedding function
        async def embed_func(texts):
            result = await emb.embed_texts(texts)
            return np.array(result)

        # 4. Cấu hình reranker (nếu enabled)
        if settings.reranker_enabled:
            async def rerank_func(query, documents, top_n=None, **kwargs):
                return await generic_rerank_api(...)
            lightrag_kwargs["rerank_model_func"] = rerank_func

        # 5. Tạo RAGAnything instance
        config = RAGAnythingConfig(
            working_dir=settings.rag_working_dir,
            enable_image_processing=settings.enable_image_processing,
            enable_table_processing=settings.enable_table_processing,
            enable_equation_processing=settings.enable_equation_processing,
            parser=settings.rag_parser,
            context_window=settings.context_window,
            context_mode=settings.context_mode,
            max_context_tokens=settings.max_context_tokens,
        )
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=llm.chat_with_image,
            embedding_func=EmbeddingFunc(
                embedding_dim=settings.embed_dim,
                max_token_size=8192,
                func=embed_func,
            ),
            lightrag_kwargs=lightrag_kwargs,
        )

        await rag._ensure_lightrag_initialized()
        return rag
```

### 6.4 API Layer — FastAPI Backend

#### 6.4.1 App Factory

```python
def create_app() -> FastAPI:
    app = FastAPI(title="RAG-Anything-UIT")

    # CORS cho Streamlit frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,  # ["http://localhost:8501"]
    )

    # Lifespan: startup/shutdown
    @asynccontextmanager
    async def lifespan(app):
        await container.startup()
        yield
        await container.shutdown()

    app.include_router(router)
    return app
```

#### 6.4.2 API Endpoints

**POST `/chat` — Truy vấn RAG:**

```python
class QueryRequest(BaseModel):
    query: str                                    # Câu hỏi
    mode: Literal["local", "global", "hybrid",
                  "naive", "mix", "bypass"] = "hybrid"
    top_k: int | None = None                      # Số kết quả truy xuất
    response_type: str = "Multiple Paragraphs"    # Định dạng trả lời
    conversation_history: list[dict] = []          # Lịch sử hội thoại
    only_need_context: bool = False               # Chỉ trả context, không sinh

class QueryResponse(BaseModel):
    answer: str             # Câu trả lời
    elapsed_seconds: float  # Thời gian xử lý
    mode: str               # Chế độ đã dùng
```

**POST `/ingest/upload` — Upload và nhập tài liệu:**

```python
@router.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    start_page: int | None = Form(None),
    end_page: int | None = Form(None),
):
    # 1. Lưu file tạm
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)

    # 2. Xử lý ingestion
    await svc.ingest_document(tmp_path, start_page=start_page, end_page=end_page)

    # 3. Dọn file tạm
    tmp_path.unlink(missing_ok=True)
```

**GET `/health` — Health Check:**

```json
{
    "status": "healthy",
    "rag_loaded": true
}
```

**GET `/system/info` — Thông tin hệ thống:**

```json
{
    "config": { ... },      // Cấu hình RAGAnything
    "processors": { ... }   // Thông tin processor
}
```

### 6.5 UI Layer — Giao diện Streamlit

Nhóm xây dựng giao diện chat bằng **Streamlit** với đầy đủ tính năng:

#### 6.5.1 Sidebar — Cấu hình

```
┌──────────────────────────┐
│  Cài đặt truy vấn        │
│                           │
│  Chế độ truy vấn:        │
│  [hybrid ▾]              │
│  hybrid = kết hợp         │
│  mix = đồ thị + vector   │
│  local = ngữ cảnh cục bộ │
│  global = tri thức toàn cục│
│                           │
│  Số kết quả (Top K):     │
│  ├──●──────────┤ 10      │
│  1              50       │
│                           │
│  Định dạng trả lời:      │
│  [Multiple Paragraphs ▾] │
│                           │
│  ─────────────────        │
│  Nhập tài liệu           │
│  [Tải tài liệu lên]     │
│  pdf, docx, pptx, xlsx,  │
│  png, jpg, txt, md        │
│  [Nhập tài liệu]        │
│                           │
│  ─────────────────        │
│  ▶ Thông tin hệ thống    │
│    [Tải thông tin]        │
└──────────────────────────┘
```

#### 6.5.2 Chat Interface

```
┌──────────────────────────────────────┐
│  📚 RAG-Anything Chat                │
│  Đặt câu hỏi về nội dung tài liệu  │
│  đã được nhập vào hệ thống!         │
│                                       │
│  ┌─────────────────────────────────┐ │
│  │ 👤 Nghiên cứu sinh cần hoàn    │ │
│  │    thành bao nhiêu tín chỉ?    │ │
│  └─────────────────────────────────┘ │
│                                       │
│  ┌─────────────────────────────────┐ │
│  │ 🤖 Theo quy chế đào tạo,      │ │
│  │    nghiên cứu sinh phải hoàn   │ │
│  │    thành tối thiểu 90 tín chỉ  │ │
│  │    trong thời gian 3 năm...    │ │
│  │                                 │ │
│  │ ⏱ 3.45s · chế độ: hybrid      │ │
│  └─────────────────────────────────┘ │
│                                       │
│  ┌─────────────────────────────────┐ │
│  │ Bạn muốn hỏi gì?         [↵]  │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
```

#### 6.5.3 Xử lý lỗi

| Lỗi | Thông báo tiếng Việt |
|---|---|
| Mất kết nối backend | "Không thể kết nối backend. Hãy chạy `python run_api.py` trước." |
| Timeout | "Yêu cầu bị quá thời gian. Hệ thống RAG mất quá lâu để phản hồi." |
| Lỗi khác | "Lỗi không mong đợi: {chi tiết lỗi}" |

---

## 7. Hỗ trợ tiếng Việt

### 7.1 Kiến trúc prompt đa ngôn ngữ

Nhóm tận dụng hệ thống **Prompt Language Registration** của RAG-Anything:

```python
# Đăng ký ngôn ngữ mới
register_prompt_language("vi", PROMPTS_VI)

# Kích hoạt ngôn ngữ
set_prompt_language("vi")
```

### 7.2 Danh sách prompt tiếng Việt

Nhóm xây dựng **bộ prompt tiếng Việt hoàn chỉnh** gồm 30+ template, phân loại theo chức năng:

#### System Prompts (Vai trò chuyên gia):
| Key | Prompt |
|---|---|
| `IMAGE_ANALYSIS_SYSTEM` | "Bạn là chuyên gia phân tích hình ảnh. Hãy cung cấp mô tả chi tiết và chính xác." |
| `TABLE_ANALYSIS_SYSTEM` | "Bạn là chuyên gia phân tích dữ liệu. Hãy cung cấp phân tích bảng chi tiết với các nhận định cụ thể." |
| `EQUATION_ANALYSIS_SYSTEM` | "Bạn là chuyên gia toán học. Hãy cung cấp phân tích toán học chi tiết." |
| `GENERIC_ANALYSIS_SYSTEM` | "Bạn là chuyên gia phân tích nội dung chuyên về {content_type}." |

#### Vision Prompts (Phân tích hình ảnh):
- `vision_prompt`: Phân tích hình ảnh đơn lẻ → JSON có `detailed_description` + `entity_info`.
- `vision_prompt_with_context`: Phân tích hình ảnh kết hợp ngữ cảnh xung quanh.

#### Table Prompts (Phân tích bảng):
- `table_prompt`: Phân tích cấu trúc bảng, xu hướng dữ liệu, phát hiện quan trọng.
- `table_prompt_with_context`: Phân tích bảng trong bối cảnh tài liệu.

#### Equation Prompts (Phân tích công thức):
- `equation_prompt`: Giải thích ý nghĩa toán học, biến số, ứng dụng.
- `equation_prompt_with_context`: Phân tích công thức trong ngữ cảnh lý thuyết.

#### Query Prompts (Truy vấn):
- `QUERY_IMAGE_DESCRIPTION`: Mô tả ngắn gọn nội dung hình ảnh.
- `QUERY_TABLE_ANALYSIS`: Phân tích dữ liệu bảng.
- `QUERY_EQUATION_ANALYSIS`: Giải thích công thức toán học.
- `QUERY_ENHANCEMENT_SUFFIX`: Yêu cầu câu trả lời toàn diện dựa trên ngữ cảnh đa phương thức.

### 7.3 Query User Prompt

Mỗi truy vấn được thêm instruction bắt buộc trả lời tiếng Việt:

```
"Always respond in Vietnamese (Tiếng Việt). Use Vietnamese terminology."
```

---

## 8. Công nghệ sử dụng

### 8.1 Bảng tổng hợp công nghệ

| Tầng | Công nghệ | Phiên bản/Chi tiết | Vai trò |
|---|---|---|---|
| **LLM Text** | Qwen3-30B-A3B | via OpenRouter | Sinh câu trả lời, trích xuất entity |
| **LLM Vision** | Qwen2.5-VL-72B-Instruct | via OpenRouter | Phân tích hình ảnh |
| **Embedding** | Qwen3-Embedding-8B | 4096 dimensions | Vector embedding |
| **Reranker** | Cohere Rerank v3.5 | via OpenRouter | Sắp xếp lại kết quả |
| **Document Parser** | MinerU | GPU/CUDA | Trích xuất nội dung PDF |
| **RAG Framework** | LightRAG + RAG-Anything | Latest | Knowledge Graph RAG |
| **Vector DB** | NanoVectorDB / Milvus Lite | JSON / SQLite | Lưu trữ vector |
| **Graph DB** | NetworkX | GraphML format | Đồ thị tri thức |
| **Backend** | FastAPI + Uvicorn | Async | REST API server |
| **Frontend** | Streamlit | Chat UI | Giao diện người dùng |
| **HTTP Client** | aiohttp | Async | Giao tiếp API |
| **Config** | Pydantic Settings | v2 | Quản lý cấu hình |
| **Language** | Python | ≥3.11 | Ngôn ngữ lập trình |

### 8.2 Lý do chọn công nghệ

**Tại sao chọn OpenRouter?**
- Truy cập nhiều mô hình qua một API key duy nhất.
- Hỗ trợ cả text, vision, embedding, và reranking.
- Chi phí hợp lý, không cần tự host mô hình.

**Tại sao chọn Qwen models?**
- Hiệu suất cao trên tiếng Việt (được huấn luyện trên dữ liệu đa ngôn ngữ).
- Qwen2.5-VL-72B là một trong những Vision LM mạnh nhất.
- Qwen3-30B-A3B (Mixture of Experts) cân bằng giữa chất lượng và tốc độ.

**Tại sao chọn LightRAG?**
- Đồ thị tri thức giúp truy vấn chính xác hơn RAG truyền thống.
- Nhiều chế độ truy vấn (local, global, hybrid) cho các loại câu hỏi khác nhau.
- Kiến trúc modular, dễ mở rộng.

**Tại sao chọn FastAPI + Streamlit?**
- FastAPI: async native, tự động sinh OpenAPI docs, type-safe.
- Streamlit: nhanh chóng xây dựng UI prototype, hỗ trợ chat interface.

---

## 9. Hướng dẫn cài đặt và sử dụng

### 9.1 Yêu cầu hệ thống

- Python ≥ 3.11
- GPU NVIDIA với CUDA (khuyến nghị cho MinerU)
- RAM ≥ 16GB
- Disk ≥ 10GB

### 9.2 Cài đặt

```bash
# 1. Clone repository
git clone <repository-url>
cd RAG-Anything-UIT

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Cấu hình environment
cp .env.example .env
# Chỉnh sửa .env: thêm OPENROUTER_API_KEY
```

### 9.3 Sử dụng

**Khởi động hệ thống:**

```bash
# Terminal 1: Khởi động API server
python run_api.py
# Server chạy tại http://localhost:8000

# Terminal 2: Khởi động UI
streamlit run run_ui.py
# UI chạy tại http://localhost:8501
```

**Nhập tài liệu qua CLI:**

```bash
# Nhập file PDF
python ingest_pdf.py path/to/document.pdf

# Nhập với giới hạn trang
python ingest_pdf.py path/to/document.pdf --start-page 0 --end-page 10

# Nhập lại từ content đã parse
python reingest.py
```

**Truy vấn qua CLI:**

```bash
# Truy vấn mặc định (hybrid mode)
python query_system.py "Câu hỏi của bạn"

# Truy vấn với tùy chọn
python query_system.py "Câu hỏi" --mode mix --top-k 15 --response-type "Bullet Points"
```

**Sử dụng qua API:**

```bash
# Health check
curl http://localhost:8000/health

# Truy vấn
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Nghiên cứu sinh cần bao nhiêu tín chỉ?", "mode": "hybrid"}'

# Upload tài liệu
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@document.pdf"
```

---

## 10. Kết luận và hướng phát triển

### 10.1 Kết quả đạt được

Nhóm đã xây dựng thành công hệ thống RAG-Anything-UIT với các thành tựu:

1. **Hệ thống RAG đa phương thức hoàn chỉnh:** Xử lý được text, hình ảnh, bảng biểu, công thức toán học trong cùng một pipeline thống nhất.

2. **Đồ thị tri thức tích hợp:** Sử dụng LightRAG để xây dựng knowledge graph, cho phép truy vấn sâu hơn so với vector search đơn thuần, với 6 chế độ truy vấn khác nhau.

3. **Hỗ trợ tiếng Việt toàn diện:** 30+ prompt template tiếng Việt, giao diện chat tiếng Việt, kết quả trả lời tiếng Việt.

4. **Kiến trúc production-ready:** API RESTful với FastAPI, dependency injection, retry logic, error handling, health check, CORS.

5. **Linh hoạt và mở rộng:** Hỗ trợ nhiều vector database backend (NanoVectorDB, Milvus), nhiều model LLM qua OpenRouter, reranker tùy chọn.

### 10.2 Hạn chế hiện tại

1. **Phụ thuộc API bên ngoài:** Toàn bộ LLM, embedding, reranking đều qua OpenRouter — phụ thuộc vào mạng và chi phí API.
2. **MinerU yêu cầu GPU:** Phân tích tài liệu cần GPU NVIDIA, hạn chế triển khai trên máy không có GPU.
3. **Chưa có streaming response:** API trả kết quả một lần, chưa hỗ trợ streaming cho trải nghiệm tốt hơn.
4. **Chưa có authentication:** API endpoints chưa có cơ chế xác thực người dùng.

### 10.3 Hướng phát triển

1. **Streaming response:** Tích hợp Server-Sent Events (SSE) để trả kết quả dần dần.
2. **Self-hosted LLM:** Triển khai mô hình LLM local bằng vLLM hoặc Ollama để giảm phụ thuộc API.
3. **Multi-user support:** Thêm authentication, authorization, và tách biệt dữ liệu theo user.
4. **Đánh giá chất lượng:** Xây dựng bộ benchmark đánh giá chất lượng truy vấn trên dữ liệu tiếng Việt.
5. **Tối ưu hiệu suất:** Cache thông minh hơn, lazy loading, và tối ưu embedding batch.

---

## Phụ lục

### A. Sơ đồ luồng dữ liệu toàn hệ thống

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          INGESTION FLOW                                       │
│                                                                               │
│  PDF/DOCX/Image ──→ MinerU Parser ──→ content_list.json                      │
│                         │                    │                                │
│                    (GPU/CUDA)           ┌────┴────┐                           │
│                                         │         │                           │
│                                     Text items  Multimodal items              │
│                                         │         │                           │
│                                         │    VLM/LLM Analysis                 │
│                                         │    (Qwen Vision/Text)               │
│                                         │         │                           │
│                                         │    Enhanced captions                │
│                                         │         │                           │
│                                         └────┬────┘                           │
│                                              │                                │
│                                         Chunking                              │
│                                         (1200 tokens)                         │
│                                              │                                │
│                                    Entity/Relation Extraction                 │
│                                         (LLM + Gleaning)                      │
│                                              │                                │
│                                         Merge & Store                         │
│                                    ┌────┬────┼────┬────┐                      │
│                                    │    │    │    │    │                       │
│                                  VDB  VDB  VDB  KG  KV                       │
│                               (ent) (rel) (chk) (GML)(JSON)                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                            QUERY FLOW                                         │
│                                                                               │
│  User Question ──→ Keyword Extraction (LLM)                                  │
│                         │                                                     │
│                    HL + LL Keywords                                            │
│                         │                                                     │
│              ┌──────────┼──────────┐                                          │
│              │          │          │                                           │
│          Local      Global     Naive                                          │
│         (entity    (relation   (chunk                                         │
│          search)    search)    search)                                         │
│              │          │          │                                           │
│              └──────────┼──────────┘                                          │
│                         │                                                     │
│                    Merge Results                                              │
│                         │                                                     │
│                    Reranking (Cohere v3.5)                                    │
│                         │                                                     │
│                    Token Truncation                                           │
│                         │                                                     │
│                    LLM Generation (Qwen3)                                    │
│                         │                                                     │
│                    Vietnamese Answer                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### B. Cấu hình môi trường (.env.example)

```env
# ── API ──────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-v1-xxxxx

# ── Models ───────────────────────────────────────
LLM_TEXT_MODEL=qwen/qwen3-30b-a3b
LLM_VLM_MODEL=qwen/qwen2.5-vl-72b-instruct
EMBED_MODEL=qwen/qwen3-embedding-8b
EMBED_DIM=4096

# ── Storage ──────────────────────────────────────
VECTOR_STORAGE=NanoVectorDBStorage
MILVUS_DB_PATH=./milvus_lite.db
MILVUS_COLLECTION=rag_multimodal_collection

# ── RAG Processing ───────────────────────────────
RAG_PARSER=mineru
PARSE_METHOD=auto
ENABLE_IMAGE_PROCESSING=true
ENABLE_TABLE_PROCESSING=true
ENABLE_EQUATION_PROCESSING=true
CONTEXT_WINDOW=1
CONTEXT_MODE=page
MAX_CONTEXT_TOKENS=2000

# ── Reranker ─────────────────────────────────────
RERANKER_ENABLED=true
RERANKER_MODEL=cohere/rerank-v3.5

# ── Language ─────────────────────────────────────
SUMMARY_LANGUAGE=Vietnamese
QUERY_USER_PROMPT=Always respond in Vietnamese (Tiếng Việt). Use Vietnamese terminology.

# ── Infrastructure ───────────────────────────────
TIMEOUT=120
MAX_RETRIES=3
LOG_LEVEL=INFO
MINERU_DEVICE=cuda
```

---

*Báo cáo được thực hiện bởi nhóm RAG-Anything-UIT — Đại học Công nghệ Thông tin, ĐHQG TP.HCM*
