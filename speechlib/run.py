import os, sys
from speechlib import Transcriptor
from speechlib import PreProcessor
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def sentiment_classifier(file):
    file_path = os.path.join("/svc/project/genaipilot/speechlib/speechlib/logs/", file)
    llm = Ollama(model="llama3:70b")

    # string_prompt = (
    #     PromptTemplate.from_template("""신용 카드 회사의 고객 서비스 전화를 분류하는 사람이 되어 주세요. 입력된 {text}로부터 SPEAKER_01과 SPEAKER_00의 대화에서 누가 상담원이고 누가 고객인지 추론하세요. 
    #                                 고객 통화를 네 가지 카테고리로 분류해야 합니다: 1. 문의, 2. 불만, 3. 칭찬, 4. 제안의 네 가지 카테고리로 분류해야 합니다. 
    #                                 대화 내용을 분석하여 위의 카테고리 중 가장 적합한 카테고리로 분류하세요. 2. 불만으로 분류된 경우 불만의 심각도를 0에서 10까지의 척도로 표시하세요.
    #                                 입력 {text}에 SPEAKER_01 또는 SPEAKER_00 중 하나만 존재하는 경우, 최대한 대화 내용에 기반하여 상담 내용을 추론하고 카테고리를 분류하되, STT의 품질이 좋지 않음을 표시해주세요. 
    #                                 모든 답변은 영어가 아닌 한국어로 작성해야 합니다.""")
    # )

    string_prompt = (
        PromptTemplate.from_template("""I want you to be the one to categorize customer service calls for a credit card company. 
                                    Infer who is an agent and who is a customer from the {text} conversation between SPEAKER_01 and SPEAKER_00. 
                                    You will need to categorize customer calls into four categories: 1. queries(문의), 2. complaints(불만), 3. compliments(칭찬), and 4. suggestions(제안). 
                                    Analyze the {text} content of the conversation and classify it into the most appropriate of the above categories. 
                                    For complaints, please indicate the severity of the complaint on a scale from 0 to 10. 
                                    
                                    Your answer must be in Korean, not English.

                                    The format of output should be like this

                                    Category: 
                                    (In case category is complaints) Complaints Score:
                                    Summary: 
                                    SPEAKER_00: (customer or agent)
                                    SPEAKER_01: (customer or agent)
                                    """)
    )

    with open(file_path, 'a+') as f:
        # print(f.tell())
        f.seek(0)
        t = f.read()
        print("-"*100)
        print(t)

        string_prompt_value = string_prompt.format_prompt(text=t)
        result = llm.invoke(string_prompt_value)
        
        if t == "":
            raise ValueError('Cannot read text file')

        f.write("\n" + result)
        print(result + "\n")

def run(path, classify):
    file = path
    voice_folder = ""
    language = "ko"
    log_folder = "logs"
    modelSize = "large-v3" 
    ACCESS_TOKEN = "hugging face access token"

    transcriptor = Transcriptor(file, log_folder, language, modelSize, ACCESS_TOKEN)
    # res = transcriptor.faster_whisper()
    res = transcriptor.whisper()

    if not res:
        raise ValueError('STT is not implemented')

    print("res", res)

    # If STT classifying needed
    if classify == "c":
        file_name = os.path.splitext(os.path.basename(file))[0]
        text = file_name + ".txt"
        sentiment_classifier(text)

if __name__ == "__main__":
    # python run.py finance/balance
    input_file, classify = sys.argv[1], sys.argv[2]
    path = os.path.join("/svc/project/genaipilot/data/", f"{input_file}")

    if os.path.isdir(path):
        print(f"{path} is a directory.")
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for f in files:
            print(path + '/' + f)
            file = path + '/' + f
            run(file, classify)
    else:
        print(f"{path} is a file.")
        run(path, classify)