# Source Documents

This folder contains the official HAN documents used as the knowledge base for
the RAG chatbot. All answers must be grounded in these files, if a question
cannot be answered from these sources, the bot refers the student to the
internship coordinator.

## Currently ingested

| File | Description | Audience | Year |
|---|---|---|---|
| `GI_manual_2025-2026.pdf` | Graduation Internship Manual: full guidelines for the Graduation internships. | IB and CS students | 2025-2026 |
| `3rd year internship manual Sem2 2025-2026` | 3rd year internship manual: full guidelines for the 3rd year internships. | IB and CS students | 2025-2026 | 

## Pending

- [ ] FAQ document? (format TBD / Word / Markdown / spreadsheet)

## Notes for ingestion

- The GI manual is text-based (not scanned), so `PyPDFLoader` will work directly, no OCR needed.
- The manual contains structured sections, tables (timeline, HandIn codes, graduation protocol), and per-specialisation appendices. Chunking should preserve section boundaries where possible (RecursiveCharacterTextSplitter with paragraph/section separators).
- Page numbers should be kept in chunk metadata so the bot can cite "Manual page X" when answering.
- The per-specialisation appendices (O&C, M&S, Finance, SCM, CS) are largely parallel in structure, chunk metadata should include the specialisation tag so retrieval can be filtered when a student mentions their stream.
