import pytng

with pytng.TNGFile('nothere.tng', 'r') as f:
    f.read()
