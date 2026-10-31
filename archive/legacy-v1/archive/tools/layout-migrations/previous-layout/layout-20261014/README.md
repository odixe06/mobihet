# Migration đã thực hiện

Scripts dùng một lần, giữ để kiểm cách chuyển folder/alias/docs. Không chạy lại trên layout đã hoàn tất.

[Manifest](../../../../../docs/archive/operations/layout-migration-2026-10-14.json) · [Cấu trúc](../../../../../docs/operations/project-layout.md).

[move_build_packages.py](move_build_packages.py) chuyển module riêng vào đúng build, giữ alias/import/hash cũ.
Đã thực hiện; lần sau dùng [check_package_layout.py](../../../../../scripts/check_package_layout.py) để kiểm, không chạy lại migration.
