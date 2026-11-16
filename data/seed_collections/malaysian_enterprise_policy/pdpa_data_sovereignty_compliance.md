# PDPA 2010 (Amendment 2024) & Enterprise Data Sovereignty Compliance Policy

## 1.1 Data Classification & Sovereign Processing Mandate

All enterprise knowledge assets ingested into internal AI workspaces must be classified under three mandatory tiers in accordance with Malaysia's Personal Data Protection Act 2010 (PDPA) and its 2024 amendments:

| Classification Tier | Examples | Permitted Inference Mode | External Transmission |
| :--- | :--- | :--- | :--- |
| `CONFIDENTIAL` | NRIC / MyKad numbers, employee salaries, medical records, sovereign hardware designs | `LOCAL_ONPREM` ONLY (Air-gapped Ollama + local Qdrant) | Strictly Prohibited (0% external API or web search) |
| `INTERNAL` | Non-PII SOPs, redacted operational policies, anonymized QA datasets | `LOCAL_ONPREM` or `CLOUD_API` (DeepSeek zero-retention mode) | Permitted only via approved TLS 1.3 API adapters |
| `PUBLIC` | Published product brochures, public press releases | `LOCAL_ONPREM` or `CLOUD_API` | Permitted |

<!-- page: 2 -->
## 2.1 Seksyen 12B: Notifikasi Insiden Kebocoran Data (*Data Breach Notification*)

Selaras dengan keperluan pindaan Akta Perlindungan Data Peribadi (PDPA) Seksyen 12B:
1. Sebarang insiden pelanggaran atau kebocoran data peribadi (*personal data breach*) yang berisiko menyebabkan kemudaratan ketara kepada subjek data wajib dilaporkan kepada **Pesuruhjaya Perlindungan Data Peribadi (PDP Commissioner)** dalam tempoh tidak melebihi **72 jam** selepas insiden disedari.
2. Subjek data yang terjejas juga perlu dimaklumkan tanpa kelewatan tidak munasabah bersama langkah pembetulan yang disyorkan.
3. **Polisi Sifar Kebocoran RAG (*Zero-Web-Fallback Policy*):** Sistem RAG perusahaan dilarang menyalurkan sebarang pertanyaan pengguna ke enjin carian web awam (seperti Tavily atau carian internet luar) apabila bukti dokumen dalaman tidak mencukupi. Sistem wajib mengeluarkan penolakan berhemah (*Graceful Refusal*) tanpa membuat andaian.
