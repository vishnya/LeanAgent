import os
from loguru import logger
from pydantic import BaseModel
from fastapi import FastAPI, Header
from github import Github, GithubIntegration
from github.GithubException import UnknownObjectException
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

APP_ID = os.environ.get("APP_ID")
if APP_ID is None:
    raise ValueError("The environment variable `APP_ID` is not set")
app_key = open("app-private-key.pem").read()
github_integration = GithubIntegration(APP_ID, app_key)


class User(BaseModel):
    login: str


class Repository(BaseModel):
    name: str
    owner: User


class PushEvent(BaseModel):
    ref: str
    repository: Repository


PR_TITLE = "[LeanCopilotBot] `sorry` Removed by Lean Copilot"


PR_BODY = """We identify the files containing theorems that have `sorry`, and replace them with a proof discovered by [Lean Copilot](https://github.com/lean-dojo/LeanCopilot).

---

<i>~LeanCopilotBot - From the [LeanDojo](https://leandojo.org/) family</i>

[:octocat: repo](https://github.com/lean-dojo/LeanCopilotBot) | [🙋🏾 issues](https://github.com/lean-dojo/LeanCopilotBot/issues) | [🏪 marketplace](https://github.com/marketplace/LeanCopilotBot)
"""


TARGET_BRANCH = "main"

TMP_BRANCH = "_LeanCopilotBot"

# TODO: remove
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/")
def handle(event: PushEvent, x_github_event: str = Header()) -> None:
    logger.info(event)
    if x_github_event == "push":
        handle_push(event)


def delete_branch(repo, branch_name: str) -> None:
    try:
        ref = repo.get_git_ref(f"heads/{branch_name}")
        ref.delete()
    except UnknownObjectException:
        pass


def handle_push(event: PushEvent) -> None:
    if event.ref != f"refs/heads/{TARGET_BRANCH}":
        logger.debug(f"Not on the `{TARGET_BRANCH}` branch")
        return

    owner = event.repository.owner.login
    repo_name = event.repository.name
    installation_id = github_integration.get_installation(owner, repo_name).id
    github_connection = Github(
        login_or_token=github_integration.get_access_token(installation_id).token
    )
    repo = github_connection.get_repo(f"{owner}/{repo_name}")

    tree = repo.get_git_tree(sha=TARGET_BRANCH, recursive=True).tree
    lean_files = [file.path for file in tree if file.path.endswith(".lean")]
    if len(lean_files) == 0:
        logger.debug("No Lean files found")
        return

    delete_branch(repo, TMP_BRANCH)
    target_branch = repo.get_branch(TARGET_BRANCH)
    repo.create_git_ref(ref=f"refs/heads/{TMP_BRANCH}", sha=target_branch.commit.sha)

    for file_path in lean_files:
        ## For testing purposes, we separate out this file.
        file_contents = repo.get_contents(file_path, ref=TARGET_BRANCH)
        new_file_contents = (
            "// This is a Lean file\n" + file_contents.decoded_content.decode()
        )
        repo.update_file(
            file_path,
            "Add comment to Lean file",
            new_file_contents,
            file_contents.sha,
            branch=TMP_BRANCH,
        )
        ## After debugging, we will replace the three lines above by calling the Lean script that meta programs the given Lean files.

    repo.create_pull(
        title=PR_TITLE,
        body=PR_BODY,
        head=TMP_BRANCH,
        base=TARGET_BRANCH,
    )
