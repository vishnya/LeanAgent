from leanagent_utils import remove_marks

def test_remove_marks_no_marks():
    assert remove_marks("hello world") == "hello world"

def test_remove_marks_with_marks():
    assert remove_marks("<a>hello</a> world") == "hello world"

def test_remove_marks_multiple_marks():
    assert remove_marks("<a>hello</a> <a>world</a>") == "hello world"

def test_remove_marks_empty_string():
    assert remove_marks("") == ""

def test_remove_marks_only_marks():
    assert remove_marks("<a></a>") == "" 