import pathlib
import sys
file_path = pathlib.Path('.env')
text = file_path.read_text(encoding='utf-8')
lines = text.splitlines()
for i, line in enumerate(lines):
    if line.startswith('AI_PROVIDER='):
        lines[i] = 'AI_PROVIDER=gemini'
        break
file_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
