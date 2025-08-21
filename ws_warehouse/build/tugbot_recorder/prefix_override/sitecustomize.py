import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/newin/Projects/warehouse/ws_warehouse/install/tugbot_recorder'
