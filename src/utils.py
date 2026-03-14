import datetime
import re
from typing import Dict, Any, Optional

def parse_option_instrument(instrument_name: str) -> Optional[Dict[str, Any]]:
    """
    Parses a typical crypto option instrument name (e.g., BTC-25MAR22-40000-C).
    Returns a dictionary with expiry date, strike price, and option type.
    """
    # Example format: BTC-24JUN22-30000-C
    parts = instrument_name.split('-')
    if len(parts) != 4:
        return None
    
    underlying = parts[0]
    expiry_str = parts[1]
    strike_str = parts[2]
    opt_type_str = parts[3]
    
    try:
        strike = float(strike_str)
        # Parse expiry - Deribit style, e.g., 25MAR22
        # Need to handle this carefully to a timestamp if possible, or leave as string to parse later.
        # usually data has timestamps natively, but we can parse this to a datetime object.
        expiry_date = datetime.datetime.strptime(expiry_str, "%d%b%y")
    except ValueError:
        return None
        
    return {
        "underlying": underlying,
        "expiry_datetime": expiry_date,
        "strike": strike,
        "option_type": "call" if opt_type_str.upper() == "C" else "put"
    }
