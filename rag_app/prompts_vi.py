"""
Vietnamese (Tiếng Việt) prompt templates for multimodal content processing.

Provides Vietnamese-language prompt templates as an alternative to the default
English templates.  Activate at process level by calling
``set_prompt_language("vi")`` from :mod:`raganything.prompt_manager`.
"""

from __future__ import annotations

from typing import Any

PROMPTS_VI: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
PROMPTS_VI["IMAGE_ANALYSIS_SYSTEM"] = (
    "Bạn là chuyên gia phân tích hình ảnh. Hãy cung cấp mô tả chi tiết và chính xác."
)
PROMPTS_VI["IMAGE_ANALYSIS_FALLBACK_SYSTEM"] = (
    "Bạn là chuyên gia phân tích hình ảnh. Hãy cung cấp phân tích chi tiết dựa trên thông tin hiện có."
)
PROMPTS_VI["TABLE_ANALYSIS_SYSTEM"] = (
    "Bạn là chuyên gia phân tích dữ liệu. Hãy cung cấp phân tích bảng chi tiết với các nhận định cụ thể."
)
PROMPTS_VI["EQUATION_ANALYSIS_SYSTEM"] = (
    "Bạn là chuyên gia toán học. Hãy cung cấp phân tích toán học chi tiết."
)
PROMPTS_VI["GENERIC_ANALYSIS_SYSTEM"] = (
    "Bạn là chuyên gia phân tích nội dung chuyên về {content_type}."
)

# ---------------------------------------------------------------------------
# Image analysis prompts
# ---------------------------------------------------------------------------
PROMPTS_VI["vision_prompt"] = """Hãy phân tích chi tiết hình ảnh này và trả lời theo cấu trúc JSON sau:

{{
    "detailed_description": "Mô tả toàn diện và chi tiết về hình ảnh, tuân theo các hướng dẫn sau:
    - Mô tả bố cục tổng thể và cách trình bày
    - Nhận diện tất cả đối tượng, con người, văn bản và yếu tố hình ảnh
    - Giải thích mối quan hệ giữa các yếu tố
    - Ghi nhận màu sắc, ánh sáng và phong cách hình ảnh
    - Mô tả các hoạt động hoặc hành động được thể hiện
    - Nếu liên quan đến biểu đồ, sơ đồ, bao gồm các chi tiết kỹ thuật
    - Luôn sử dụng tên cụ thể thay vì đại từ",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "Tóm tắt ngắn gọn về nội dung hình ảnh và tầm quan trọng của nó (không quá 100 từ)"
    }}
}}

Thông tin bổ sung:
- Đường dẫn hình ảnh: {image_path}
- Chú thích: {captions}
- Ghi chú: {footnotes}

Hãy tập trung cung cấp phân tích hình ảnh chính xác, chi tiết để hỗ trợ truy xuất tri thức."""

PROMPTS_VI[
    "vision_prompt_with_context"
] = """Hãy phân tích chi tiết hình ảnh này kết hợp với ngữ cảnh xung quanh, và trả lời theo cấu trúc JSON sau:

{{
    "detailed_description": "Mô tả toàn diện và chi tiết về hình ảnh, tuân theo các hướng dẫn sau:
    - Mô tả bố cục tổng thể và cách trình bày
    - Nhận diện tất cả đối tượng, con người, văn bản và yếu tố hình ảnh
    - Giải thích mối quan hệ giữa các yếu tố và liên hệ với ngữ cảnh
    - Ghi nhận màu sắc, ánh sáng và phong cách hình ảnh
    - Mô tả các hoạt động hoặc hành động được thể hiện
    - Nếu liên quan đến biểu đồ, sơ đồ, bao gồm các chi tiết kỹ thuật
    - Khi phù hợp, dẫn chiếu đến mối liên hệ với nội dung xung quanh
    - Luôn sử dụng tên cụ thể thay vì đại từ",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "Tóm tắt ngắn gọn về nội dung hình ảnh, tầm quan trọng và mối liên hệ với nội dung xung quanh (không quá 100 từ)"
    }}
}}

Ngữ cảnh nội dung xung quanh:
{context}

Chi tiết hình ảnh:
- Đường dẫn hình ảnh: {image_path}
- Chú thích: {captions}
- Ghi chú: {footnotes}

Hãy tập trung cung cấp phân tích hình ảnh chính xác, chi tiết kết hợp ngữ cảnh để hỗ trợ truy xuất tri thức."""

PROMPTS_VI["text_prompt"] = """Dựa trên thông tin hình ảnh sau đây, hãy cung cấp phân tích:

Đường dẫn hình ảnh: {image_path}
Chú thích: {captions}
Ghi chú: {footnotes}

{vision_prompt}"""

# ---------------------------------------------------------------------------
# Table analysis prompts
# ---------------------------------------------------------------------------
PROMPTS_VI["table_prompt"] = """Hãy phân tích nội dung bảng này và trả lời theo cấu trúc JSON sau:

{{
    "detailed_description": "Phân tích toàn diện về bảng, bao gồm:
    - Cấu trúc và cách tổ chức của bảng
    - Tiêu đề các cột và ý nghĩa của chúng
    - Các điểm dữ liệu và mẫu hình quan trọng
    - Nhận định thống kê và xu hướng
    - Mối quan hệ giữa các yếu tố dữ liệu
    - Tầm quan trọng của dữ liệu được trình bày
    Luôn sử dụng tên cụ thể và giá trị số liệu thay vì tham chiếu chung chung.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "Tóm tắt ngắn gọn về mục đích của bảng và các phát hiện chính (không quá 100 từ)"
    }}
}}

Thông tin bảng:
Đường dẫn hình ảnh: {table_img_path}
Tiêu đề: {table_caption}
Nội dung: {table_body}
Ghi chú: {table_footnote}

Hãy tập trung trích xuất các nhận định và mối quan hệ có ý nghĩa từ dữ liệu bảng."""

PROMPTS_VI[
    "table_prompt_with_context"
] = """Hãy phân tích nội dung bảng này kết hợp với ngữ cảnh xung quanh, và trả lời theo cấu trúc JSON sau:

{{
    "detailed_description": "Phân tích toàn diện về bảng, bao gồm:
    - Cấu trúc và cách tổ chức của bảng
    - Tiêu đề các cột và ý nghĩa của chúng
    - Các điểm dữ liệu và mẫu hình quan trọng
    - Nhận định thống kê và xu hướng
    - Mối quan hệ giữa các yếu tố dữ liệu
    - Tầm quan trọng của dữ liệu trong ngữ cảnh nội dung xung quanh
    - Cách bảng hỗ trợ hoặc minh họa các khái niệm trong nội dung xung quanh
    Luôn sử dụng tên cụ thể và giá trị số liệu thay vì tham chiếu chung chung.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "Tóm tắt ngắn gọn về mục đích của bảng, phát hiện chính và mối liên hệ với nội dung xung quanh (không quá 100 từ)"
    }}
}}

Ngữ cảnh nội dung xung quanh:
{context}

Thông tin bảng:
Đường dẫn hình ảnh: {table_img_path}
Tiêu đề: {table_caption}
Nội dung: {table_body}
Ghi chú: {table_footnote}

Hãy tập trung trích xuất các nhận định và mối quan hệ có ý nghĩa từ dữ liệu bảng trong bối cảnh ngữ cảnh xung quanh."""

# ---------------------------------------------------------------------------
# Equation analysis prompts
# ---------------------------------------------------------------------------
PROMPTS_VI["equation_prompt"] = """Hãy phân tích công thức toán học này và trả lời theo cấu trúc JSON sau:

{{
    "detailed_description": "Phân tích toàn diện về công thức, bao gồm:
    - Ý nghĩa toán học và cách diễn giải
    - Các biến và định nghĩa của chúng
    - Các phép toán và hàm được sử dụng
    - Lĩnh vực ứng dụng và bối cảnh
    - Ý nghĩa vật lý hoặc lý thuyết
    - Mối quan hệ với các khái niệm toán học khác
    - Ứng dụng thực tế hoặc trường hợp sử dụng
    Luôn sử dụng thuật ngữ toán học chính xác.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "Tóm tắt ngắn gọn về mục đích và tầm quan trọng của công thức (không quá 100 từ)"
    }}
}}

Thông tin công thức:
Công thức: {equation_text}
Định dạng: {equation_format}

Hãy tập trung cung cấp nhận định toán học và giải thích tầm quan trọng của công thức."""

PROMPTS_VI[
    "equation_prompt_with_context"
] = """Hãy phân tích công thức toán học này kết hợp với ngữ cảnh xung quanh, và trả lời theo cấu trúc JSON sau:

{{
    "detailed_description": "Phân tích toàn diện về công thức, bao gồm:
    - Ý nghĩa toán học và cách diễn giải
    - Định nghĩa các biến trong ngữ cảnh
    - Các phép toán và hàm được sử dụng
    - Lĩnh vực ứng dụng và bối cảnh dựa trên tài liệu xung quanh
    - Ý nghĩa vật lý hoặc lý thuyết
    - Mối quan hệ với các khái niệm toán học khác được đề cập trong ngữ cảnh
    - Ứng dụng thực tế hoặc trường hợp sử dụng
    - Cách công thức liên hệ với cuộc thảo luận hoặc khung lý thuyết rộng hơn
    Luôn sử dụng thuật ngữ toán học chính xác.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "Tóm tắt ngắn gọn về mục đích, tầm quan trọng và vai trò trong ngữ cảnh của công thức (không quá 100 từ)"
    }}
}}

Ngữ cảnh nội dung xung quanh:
{context}

Thông tin công thức:
Công thức: {equation_text}
Định dạng: {equation_format}

Hãy tập trung cung cấp nhận định toán học và giải thích tầm quan trọng của công thức trong bối cảnh rộng hơn."""

# ---------------------------------------------------------------------------
# Generic content analysis prompts
# ---------------------------------------------------------------------------
PROMPTS_VI["generic_prompt"] = """Hãy phân tích nội dung {content_type} này và trả lời theo cấu trúc JSON sau:

{{
    "detailed_description": "Phân tích toàn diện về nội dung, bao gồm:
    - Cấu trúc và cách tổ chức nội dung
    - Thông tin và yếu tố chính
    - Mối quan hệ giữa các thành phần
    - Bối cảnh và tầm quan trọng
    - Chi tiết liên quan đến truy xuất tri thức
    Luôn sử dụng thuật ngữ chuyên ngành phù hợp với nội dung {content_type}.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "Tóm tắt ngắn gọn về mục đích và điểm chính của nội dung (không quá 100 từ)"
    }}
}}

Nội dung: {content}

Hãy tập trung trích xuất thông tin có ý nghĩa phục vụ truy xuất tri thức."""

PROMPTS_VI[
    "generic_prompt_with_context"
] = """Hãy phân tích nội dung {content_type} này kết hợp với ngữ cảnh xung quanh, và trả lời theo cấu trúc JSON sau:

{{
    "detailed_description": "Phân tích toàn diện về nội dung, bao gồm:
    - Cấu trúc và cách tổ chức nội dung
    - Thông tin và yếu tố chính
    - Mối quan hệ giữa các thành phần
    - Bối cảnh và tầm quan trọng liên quan đến nội dung xung quanh
    - Cách nội dung này liên hệ hoặc hỗ trợ cuộc thảo luận rộng hơn
    - Chi tiết liên quan đến truy xuất tri thức
    Luôn sử dụng thuật ngữ chuyên ngành phù hợp với nội dung {content_type}.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "Tóm tắt ngắn gọn về mục đích, điểm chính và mối liên hệ với ngữ cảnh xung quanh (không quá 100 từ)"
    }}
}}

Ngữ cảnh nội dung xung quanh:
{context}

Nội dung: {content}

Hãy tập trung trích xuất thông tin có ý nghĩa phục vụ truy xuất tri thức và hiểu vai trò của nội dung trong bối cảnh rộng hơn."""

# ---------------------------------------------------------------------------
# Chunk templates
# ---------------------------------------------------------------------------
PROMPTS_VI["image_chunk"] = """
Phân tích nội dung hình ảnh:
Đường dẫn hình ảnh: {image_path}
Chú thích: {captions}
Ghi chú: {footnotes}

Phân tích hình ảnh: {enhanced_caption}"""

PROMPTS_VI["table_chunk"] = """Phân tích bảng:
Đường dẫn hình ảnh: {table_img_path}
Tiêu đề: {table_caption}
Cấu trúc: {table_body}
Ghi chú: {table_footnote}

Phân tích: {enhanced_caption}"""

PROMPTS_VI["equation_chunk"] = """Phân tích công thức toán học:
Công thức: {equation_text}
Định dạng: {equation_format}

Phân tích toán học: {enhanced_caption}"""

PROMPTS_VI["generic_chunk"] = """Phân tích nội dung {content_type}:
Nội dung: {content}

Phân tích: {enhanced_caption}"""

# ---------------------------------------------------------------------------
# Query-related prompts
# ---------------------------------------------------------------------------
PROMPTS_VI["QUERY_IMAGE_DESCRIPTION"] = (
    "Hãy mô tả ngắn gọn nội dung chính, các yếu tố quan trọng và thông tin đáng chú ý trong hình ảnh này."
)

PROMPTS_VI["QUERY_IMAGE_ANALYST_SYSTEM"] = (
    "Bạn là chuyên gia phân tích hình ảnh, có khả năng mô tả chính xác nội dung hình ảnh."
)

PROMPTS_VI["QUERY_TABLE_ANALYSIS"] = """Hãy phân tích nội dung chính, cấu trúc và thông tin quan trọng của bảng dữ liệu sau:

Dữ liệu bảng:
{table_data}

Tiêu đề bảng: {table_caption}

Hãy tóm tắt ngắn gọn nội dung chính, đặc điểm dữ liệu và các phát hiện quan trọng."""

PROMPTS_VI["QUERY_TABLE_ANALYST_SYSTEM"] = (
    "Bạn là chuyên gia phân tích dữ liệu, có khả năng phân tích chính xác dữ liệu bảng."
)

PROMPTS_VI["QUERY_EQUATION_ANALYSIS"] = """Hãy giải thích ý nghĩa và công dụng của công thức toán học sau:

Công thức LaTeX: {latex}
Tiêu đề công thức: {equation_caption}

Hãy giải thích ngắn gọn ý nghĩa toán học, phạm vi ứng dụng và tầm quan trọng của công thức này."""

PROMPTS_VI["QUERY_EQUATION_ANALYST_SYSTEM"] = (
    "Bạn là chuyên gia toán học, có khả năng giải thích rõ ràng các công thức toán học."
)

PROMPTS_VI[
    "QUERY_GENERIC_ANALYSIS"
] = """Hãy phân tích nội dung loại {content_type} sau đây và trích xuất thông tin chính cùng các đặc điểm quan trọng:

Nội dung: {content_str}

Hãy tóm tắt ngắn gọn các đặc điểm chính và thông tin quan trọng của nội dung này."""

PROMPTS_VI["QUERY_GENERIC_ANALYST_SYSTEM"] = (
    "Bạn là chuyên gia phân tích nội dung, có khả năng phân tích chính xác nội dung loại {content_type}."
)

PROMPTS_VI["QUERY_ENHANCEMENT_SUFFIX"] = (
    "\n\nHãy cung cấp câu trả lời toàn diện dựa trên truy vấn của người dùng và thông tin nội dung đa phương thức được cung cấp."
)
