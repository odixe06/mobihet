# Skill quy trình thí nghiệm và tổ chức build

Đã tạo hai bản đồng nội dung quy trình: [Codex](../../.agents/skills/federated-experiment-workflow/SKILL.md)
và [Claude Code](../../.claude/skills/federated-experiment-workflow/SKILL.md).

Skill bao gồm: resume sau crash → inventory/status → restore/pull chọn lọc → exact artifact audit → cổng utility/cost/byte
→ dự báo cải tiến → notebook checkpoint/W&B/quota → tài liệu theo phương pháp/build → index/link/hash/Git/DVC → handoff.
Skill không cấp quyền train/push/backup/delete; lấy authorization và scientific settings từ session/CONTEXT.

Codex: YAML name/description, `agents/openai.yaml` với interface/default prompt, references; kiểm bằng bundled
`quick_validate.py` của skill-creator. Claude Code: project `.claude/skills/<name>/SKILL.md`, YAML name/description,
references nạp theo nhu cầu, không thêm Codex UI metadata. Hai SKILL.md và reference giữ byte parity.

Cấu trúc Claude Code/frontmatter đối chiếu [tài liệu Anthropic chính thức](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview).
Codex đóng gói theo skill-creator và `openai_yaml.md` đang cài local, không suy metadata của host này áp dụng cho host khác.
Validator PASS là kiểm packaging, không chứng minh mọi quyết định nghiên cứu sẽ đúng.
