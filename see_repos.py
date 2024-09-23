from dynamic_database import *

RAID_DIR = "/data/yingzi_ma/lean_project"
dynamic_database_json_path = RAID_DIR + "/dynamic_database_14_original.json"
db = DynamicDatabase.from_json(dynamic_database_json_path)
db.print_database_contents()
# db.delete_repository("https://github.com/AlexKontorovich/PrimeNumberTheoremAnd", "89bf7b5e3a226525e8580bae21ef543604f99b21")
# db.print_database_contents()
# RAID_DIR = "/data/yingzi_ma/lean_project"
# db.to_json(dynamic_database_json_path)