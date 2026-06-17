import pandas as pd

VIOLATION_SEVERITY = {
    "DOUBLE PARKING": 10,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 9,
    "PARKING IN A MAIN ROAD": 8,
    "PARKING NEAR ROAD CROSSING": 8,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 7,
    "NO PARKING": 5,
    "WRONG PARKING": 4,
    "DEFECTIVE NUMBER PLATE": 1,
}

VEHICLE_SIZE = {
    "HGV": 10, "TANKER": 10,
    "BUS": 9,
    "MAXI-CAB": 7, "LGV": 7,
    "VAN": 6,
    "CAR": 5, "GOODS AUTO": 5,
    "PASSENGER AUTO": 4,
    "MOTOR CYCLE": 2, "SCOOTER": 2,
    "MOPED": 1,
}


def parse_violations(v):
    if pd.isna(v):
        return []
    try:
        items = str(v).strip("[]").replace('""', "").replace('"', "").split(",")
        return [item.strip() for item in items if item.strip()]
    except Exception:
        return []


def get_max_severity(violations_list):
    if not violations_list:
        return 3
    return max(VIOLATION_SEVERITY.get(v, 3) for v in violations_list)


def get_vehicle_size(vtype):
    if pd.isna(vtype):
        return 3
    return VEHICLE_SIZE.get(str(vtype).strip().upper(), 3)


def compute_time_factor(hour, is_weekend):
    if hour in (8, 9, 10) or hour in (17, 18, 19):
        factor = 3.0
    elif hour in (7, 11, 16, 20):
        factor = 2.0
    elif 11 <= hour <= 16:
        factor = 1.5
    elif 21 <= hour <= 23:
        factor = 1.0
    else:
        factor = 0.5
    if is_weekend:
        factor *= 0.7
    return factor
