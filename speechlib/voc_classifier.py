import pandas as pd
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from natsort import natsorted

def save_sheets_as_text(excel_file_path, output_dir):
    # 엑셀 파일 읽기
    xls = pd.ExcelFile(excel_file_path)
    
    # 각 시트를 텍스트 파일로 저장
    for sheet_name in xls.sheet_names:
        # 시트 읽기
        df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=1)

        selected_columns = df.iloc[:, [2, 5]]
        
        # 텍스트 파일 경로 설정
        text_file_path = os.path.join(output_dir, f"{sheet_name}.txt")
        
        # DataFrame을 텍스트 파일로 저장
        selected_columns.to_csv(text_file_path, sep='\t', index=False)
        print(f"Saved {sheet_name} to {text_file_path}")

def read_file_with_extra_columns(file_path, sep='\t'):
    data = []
    max_columns = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            fields = line.strip().split(sep)
            data.append(fields)
            if len(fields) > max_columns:
                max_columns = len(fields)
    
    # Create a DataFrame with the maximum number of columns found
    columns = [f'Column{i+1}' for i in range(max_columns)]
    df = pd.DataFrame(data, columns=columns)
    
    return df

def text_to_excel(input_dir, output_file, sep='\t'):
    # Create a Pandas Excel writer using openpyxl as the engine.
    writer = pd.ExcelWriter(output_file, engine='openpyxl')
    
    # Get a list of all text files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]    
    # files.sort()
    files = natsorted(files)
    
    for filename in files:
        print(filename)
        file_path = os.path.join(input_dir, filename)
        
        try:
            # Read the text file into a DataFrame
            df = read_file_with_extra_columns(file_path, sep=sep)
        except pd.errors.ParserError as e:
            print(f"Error parsing {file_path}: {e}")
            continue
    
        
        # Get the sheet name from the file name (without extension)
        sheet_name = os.path.splitext(filename)[0]
        
        # Write the DataFrame to the Excel file
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Saved {filename} to sheet {sheet_name}")

    # Save the Excel file
    writer.close()
    print(f"Excel file saved to {output_file}")


def classifier(file):
    directory = "/svc/project/genaipilot/speechlib/speechlib/code_text/"
    file_path = os.path.join(directory, file)
    llm = Ollama(model="llama3.1:70b")


    string_prompt = (
        PromptTemplate.from_template("""
                                    I want you to categorize customer service calls {text}, evaluate the quality of calls and 
                                    predict the probabilty of the calls being filed as a Financial Supervisory Service report for a credit card company. 
                                    You need to categorize customer calls into 30 categories below: 

                                    Analyze the content of the conversation and classify it into the most appropriate of the above categories. 
                                    If a conversation is classified into more than one category, enumerate categories and proportion of each. 
                                    Then explain the reason why you categorized.

                                    After classifying, please rate the satisfaction of customer service on a scale from 1 to 10. 
                                    Then explain why you evaluate the satisfaction score. If the satisfaction score is below 5, 
                                    then predict the probability of this inquiry being filed as a Financial Supervisory Service report 
                                    based on satisfaction score. You should explain your rationale.

                                    The format of output should be like a format below. Every response must be in Korean. 한글로 답하시오.
                                    -------------------------------------------------------
       
                                    """)
    )

    with open(file_path, 'a+') as f:
        # print(f.tell())
        f.seek(0)
        t = f.read()

        string_prompt_value = string_prompt.format_prompt(text=t)
        result = llm.invoke(string_prompt_value)
        
        print(t)
        print("-"*100)
        print(result)
        print("-"*100)
        
        if t == "":
            raise ValueError('Cannot read text file')

        f.write("\n" + result)
        print(result + "\n")

if __name__ == "__main__":
    # directory = "/svc/project/genaipilot/speechlib/speechlib/코드별상담내역/2_CRM문의"
    # files = os.listdir(directory)
    # for file in files:
    #     save_sheets_as_text(f"{directory}/{file}", "/svc/project/genaipilot/speechlib/speechlib/code_text")

    directory = "/svc/project/genaipilot/speechlib/speechlib/code_text"
    # files = os.listdir(directory)
    # for file in files:
    #     print(file)
    #     classifier(file)

    text_to_excel(directory, directory + '/results.xlsx')