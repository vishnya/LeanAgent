from lean_dojo import *

url = "https://github.com/teorth/pfr"
commit = "6a5082ee465f9e44cea479c7b741b3163162bb7e"
repo = LeanGitRepo(url, commit)
traced_repo = trace(repo)