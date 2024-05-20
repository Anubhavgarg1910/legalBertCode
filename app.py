def legal_assistant(question):
    # Define a dictionary of example contexts for different types of questions
    example_contexts = {
        "murder": "In many legal jurisdictions, murder is considered a serious criminal offense punishable by law.",
        "stealing": "Stealing, theft, and robbery are illegal acts punishable by law in most legal systems.",
        "contract": "A contract is a legally binding agreement between two or more parties.",
        "civil law": "Civil law deals with disputes between individuals or organizations, often seeking compensation or resolution through the courts."
    }

    # Check if the question contains keywords related to specific legal topics
    for topic, context in example_contexts.items():
        if topic in question.lower():
            # Tokenize question and example context
            inputs = tokenizer(question, context, return_tensors="pt")

            # Perform question answering
            start_scores, end_scores = model(**inputs).values()

            # Get answer span
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores) + 1
            answer_tokens = inputs["input_ids"][0][start_index:end_index]
            answer = tokenizer.decode(answer_tokens)
            return answer

    # If no specific topic is detected, return a default response
    return "I'm sorry, I'm not sure how to answer that legal question."
