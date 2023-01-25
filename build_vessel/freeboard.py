"""
This module imports the rules and returns the required output.

minimum freeboard Chapter III - Freeboard and Stability
Type 'A' ships

  (2) A type 'A' ship is one which:

    (a) is designed to carry only liquid cargoes in bulk;

    (b) has a high integrity of the exposed deck with only small access openings to cargo compartments, closed by watertight gasketed covers of steel or equivalent material; and

    (c) has low permeability of loaded cargo compartments.

Type 'B' ships

  (5) All ships which do not come within the provisions regarding type 'A' ships in paragraphs (2) and (3) shall be considered as type 'B' ships.


Author:     Dorus Boogaard
"""
import pandas as pd
from pathlib import Path

p = Path(__file__)
relative_path = p.parent / "Rules" / "min_freeboard.csv"
free_board = pd.read_csv(relative_path)

def min_freeboard(loa, type: str ='B'):
    """
    If length is not in the table, return 6.0 meters.
    Args:
      loa (int): length overall in meters
      type (str): 'A' or 'B'
    Returns:
      float: minimum freeboard in meters
    """
    try:
      if type == 'A':
          return (free_board.loc[free_board["length"] == int(loa)].values[0][1] / 1000)
      elif type == 'B':
          return (free_board.loc[free_board["length"] == int(loa)].values[0][2] / 1000)
      else:
          raise ValueError("type should be 'A' or 'B'")
    except IndexError:
      return 6.0
        
