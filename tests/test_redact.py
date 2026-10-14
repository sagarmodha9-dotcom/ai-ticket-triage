from redact import redact


def test_hides_email_phone_ip_ssn():
    text = "I'm jane.doe@hospital.org, call 301-555-0142, my IP is 10.0.3.17, SSN 123-45-6789"
    out, counts = redact(text)
    assert "jane.doe@hospital.org" not in out
    assert "301-555-0142" not in out
    assert "10.0.3.17" not in out
    assert "123-45-6789" not in out
    assert counts == {"email": 1, "phone": 1, "ip": 1, "ssn": 1}


def test_leaves_normal_text_alone():
    text = "Outlook keeps crashing when I open the calendar"
    out, counts = redact(text)
    assert out == text
    assert counts == {}
