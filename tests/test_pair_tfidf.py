from frauddistill.student.pair_tfidf import PairTfidfDetector


def test_pair_modes_have_same_feature_shape_and_interactions():
    rows = [{"user_query": "phishing code", "target_model_answer": "send verification code in step 1"}, {"user_query": "prevent phishing", "target_model_answer": "do not send verification code; report it"}, {"user_query": "normal recipe", "target_model_answer": "boil water"}, {"user_query": "impersonate", "target_model_answer": "use this script"}]
    model = PairTfidfDetector(max_features=100).fit(rows, ["unsafe", "safe", "safe", "unsafe"])
    assert model.features(rows, "q_only").shape == model.features(rows, "y_only").shape == model.features(rows, "q_y").shape
    assert model.features(rows, "q_y").nnz > model.features(rows, "q_only").nnz
