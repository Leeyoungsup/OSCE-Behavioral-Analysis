from openai import OpenAI
import json
import os
from dotenv import load_dotenv
load_dotenv()
GptApiKey=os.getenv("OPENAI_API_KEY")

class chat(OpenAI):
    def __init__(self):
        self.client = OpenAI(api_key=GptApiKey)
        self.file_path = "./prompt.txt"
    
        # Open the file and read its content
        with open(self.file_path, "r",encoding='utf-8') as file:
            self.file_content = file.read()
    def response_data(self,text):
        response = self.client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text":self.file_content
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": text
                }
            ]
            }
        ],
        response_format={
            "type": "text"
        },
        temperature=1.0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        
        result=response.choices[0].message.content
        return result
# if __name__ == "__main__":
#     chat = chat()
#     response=chat.response_data("김수찬 김수찬 258013 258013 남성 22세 남성 22세 소판액 소판액 2ml에 75g 2ml에 75mg PRN PRN im im 소파넥 소파넥 2ml에 75mg 2ml에 75mg im im 안녕하세요. 환자분 담당 간호사 문자은입니다. 환자분 성함이 어떻게 되세요? 네 제가 한번 더 확인해 드리 겠습니다. 김수찬 김수찬 남성 22세 남성 22세 258013 258013 환자분 어제 수술 후에 통증이 있으셔서 진통제 나드리러 왔습니다. 진통제를 맞으신 후에는 통증이 조금 완화되실 수 있습니다. 혹시 진통제를 맞은 후에 어지러움과 같은 이상 증세가 나타난다면 저한테 말씀해  주세요. 혹시 더 궁금한 사항이 있으실까요? 그럼 커튼 한번 쳐드리겠습니다. 환자분 주사 맞기 전에 주사 부위 한번 확인하겠습니다 하니 잠깐 들출게요. 네 이쪽 삼각근 부위에 주사 맞으시겠습니다. 주사 맞기 전에 왼쪽으로 한번 돌아놓으실게요. 소독하겠습니다. 살짝 차 가워요. 환자분 주사 들어갑니다. 따끔해요. 내관을 당겨 혈액이 나오지 않는 것을 확인한 후에 약물을 천천히 주입합니다. 환자분 약물은 다 들어갔고 마사지 해드릴게요. 환자분 환의 정리해 드리겠습니다. 자세도 편안하게 다시 바꿔드릴게요. 환자분 이제 진통제 맞으셔서 통증이 조금 덜해지실 겁니다. 혹시 더 궁금한 사항이 있거나 불편하신 점이 있으시다면 저한테 말씀해 주세요. 커튼 다시 쳐드리겠습니다. 안녕히 계세요. 사용한 물품을 정리합니다. 간호 기록지에 대상자 명, 양명, 약의 용량 투여 경로 투여 시간 필요 시 투약의 목적, 대 상자의 반응, 투약 사유 또는 투약하지 못한 사유에 대해서 기록합니다. 이상입니다.")
    
#     file_path = "output.json"
#     json_data = json.loads(response)
#     # Write the JSON content to the file
#     with open(file_path, "w", encoding="utf-8") as file:
#         json.dump(json_data, file, ensure_ascii=False, indent=4)
#     file_path  # Return the file path to confirm save location