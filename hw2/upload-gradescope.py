import os

from dotenv import load_dotenv

from gradescopeapi.classes.connection import GSConnection
from gradescopeapi.classes.upload import upload_assignment

# load .env file
load_dotenv()

GRADESCOPE_EMAIL = os.getenv("GRADESCOPE_EMAIL")
GRADESCOPE_PASSWORD = os.getenv("GRADESCOPE_PASSWORD")
GRADESCOPE_COURSE_ID = os.getenv("GRADESCOPE_COURSE_ID")
GRADESCOPE_ASSIGNMENT_ID = os.getenv("GRADESCOPE_ASSIGNMENT_ID")

def new_session():
    connection = GSConnection()
    connection.login(GRADESCOPE_EMAIL, GRADESCOPE_PASSWORD)

    return connection.session

def send_assignment():
    # create test session
    session = new_session()

    print(f"Uploading assignment to course {GRADESCOPE_COURSE_ID} and assignment {GRADESCOPE_ASSIGNMENT_ID}")

    course_id = GRADESCOPE_COURSE_ID
    assignment_id = GRADESCOPE_ASSIGNMENT_ID

    with (
        open("delivery/gpt_solution.py", "rb") as file1,
        open("delivery/lstm_solution.py", "rb") as file2,
        open("delivery/trainer_solution.py", "rb") as file3,
    ):
        submission_link = upload_assignment(
            session,
            course_id,
            assignment_id,
            file1, file2, file3
        )

    assert submission_link is not None

if __name__ == "__main__":
    send_assignment()
