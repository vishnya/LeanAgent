from dynamic_database import *

RAID_DIR = "/data/yingzi_ma/lean_project"
dynamic_database_json_path = RAID_DIR + "/dynamic_database_PT_single_repo_no_ewc_curriculum_full.json"
db = DynamicDatabase.from_json(dynamic_database_json_path)
db.print_database_contents()
db.delete_repository("https://github.com/AlexKontorovich/PrimeNumberTheoremAnd", "29baddd685660b5fedd7bd67f9916ae24253d566")
db.print_database_contents()
RAID_DIR = "/data/yingzi_ma/lean_project"
db.to_json(dynamic_database_json_path)