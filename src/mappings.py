REGION_MAPPING = {
        "Brooklyn": 1,
        "Queens": 2,
        "New Jersey": 3,
        "West Side Highway": 4,
        "FDR Drive": 5,
        "West 60th St": 6,
        "East 60th St": 7
    }

GROUP_MAPPING = {
    "Brooklyn Bridge": 1,
    "Hugh L. Carey Tunnel": 2,
    "Williamsburg Bridge": 3,
    "Manhattan Bridge": 4,
    "Queensboro Bridge": 5,
    "Queens Midtown Tunnel": 6,
    "Holland Tunnel": 7,
    "Lincoln Tunnel": 8,
    "West Side Highway at 60th St": 9,
    "FDR Drive at 60th St": 10,
    "West 60th St": 11,
    "East 60th St": 12
}

GROUP_TO_REGION = {
    1: 1,   # Brooklyn Bridge -> Brooklyn
    2: 1,   # Hugh L. Carey Tunnel -> Brooklyn
    3: 1,   # Williamsburg Bridge -> Brooklyn
    4: 1,   # Manhattan Bridge -> Brooklyn

    5: 2,   # Queensboro Bridge -> Queens
    6: 2,   # Queens Midtown Tunnel -> Queens

    7: 3,   # Holland Tunnel -> New Jersey
    8: 3,   # Lincoln Tunnel -> New Jersey

    9: 4,   # West Side Highway at 60th St -> West Side Highway
    10: 5,  # FDR Drive at 60th St -> FDR Drive
    11: 6,  # West 60th St -> West 60th St
    12: 7   # East 60th St -> East 60th St
}