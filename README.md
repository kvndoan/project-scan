import documentation:
numpy -> https://numpy.org/doc/
spectacularAI -> https://spectacularai.github.io/docs/sdk/

first time
```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

subsequent
```
.\.venv\Scripts\Activate.ps1
```

current tests
```
python scripts/scan.py "room test"
python scripts/scan.py "low light test" --low-light
```

to view zone from tests
```
$zone = Get-ChildItem .\zones -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
    
python scripts/view_zone.py $zone.FullName
```