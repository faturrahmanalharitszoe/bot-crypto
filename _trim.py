content = open('bot/web_dashboard.py', 'r', encoding='utf-8').read()
end_marker = '</body></html>"""'
idx = content.find(end_marker)
if idx >= 0:
    trimmed = content[:idx + len(end_marker)] + '\n'
    open('bot/web_dashboard.py', 'w', encoding='utf-8').write(trimmed)
    print('Trimmed OK. Lines:', len(trimmed.splitlines()))
else:
    print('Marker not found!')
