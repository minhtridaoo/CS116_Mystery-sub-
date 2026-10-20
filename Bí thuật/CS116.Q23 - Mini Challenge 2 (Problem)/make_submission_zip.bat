@echo off
set "ROOT=C:\Users\Public\Documents\NM2~1\HCK2~1\CS116\CS0B3D~1.Q23"
pushd "%ROOT%"
if exist submission.zip del /f /q submission.zip
type nul > submission.zip
"C:\Users\ADMIN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" make_submission_zip.py
popd
