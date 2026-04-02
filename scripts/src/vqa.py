import torch
from transformers import (
    AutoProcessor,
    BlipForQuestionAnswering,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    Qwen2VLForConditionalGeneration, 
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info
from PIL import Image
import re


class BlipVqaInterpreter:
    def __init__(self, device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "Salesforce/blip-vqa-capfilt-large"
        )
        model_slug = "Salesforce/blip-vqa-capfilt-large"
        self.model = BlipForQuestionAnswering.from_pretrained(model_slug)
        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()

    def predict(self, img: Image.Image, question: str) -> str:
        encoding = self.processor(img, question, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model.generate(**encoding)  # type: ignore

        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def __call__(self, image: Image.Image, question: str) -> str:
        return self.predict(image, question)

    def complex_query(self, image: Image.Image, question: str) -> str:
        return self.predict(image, question)
    
    def verify_property(
        self, image: Image.Image, object_name: str, property_: str
    ) -> bool:
        is_plural = object_name.endswith("s")
        if is_plural:
            template = "Are the {object_name} {property}?"
        else:
            template = "Is the {object_name} {property}?"
        question = template.format(object_name=object_name, property=property_)
        inputs = self.processor(
            images=[image], text=[question], return_tensors="pt"
        ).to(self.device)

        # get 'yes' and 'no' 
        yes_label = self.processor(text=["yes"], return_tensors="pt").to(self.device)
        no_label = self.processor(text=["no"], return_tensors="pt").to(self.device)

        with torch.no_grad():
            loss_yes = self.model(**inputs, labels=yes_label.input_ids).loss.item()  # type: ignore
            loss_no = self.model(**inputs, labels=no_label.input_ids).loss.item()  # type: ignore

        return loss_yes < loss_no

    def best_description_from_options(
        self, image: Image.Image, object_name: str, property_list: list
    ) -> str:
        is_plural = object_name.endswith("s")
        if is_plural:
            template = "Are the {object_name} {property}?"
        else:
            template = "Is the {object_name} {property}?"

        loss_list = []
        for property_ in property_list:
            question = template.format(object_name=object_name, property=property_)
            inputs = self.processor(
                images=[image], text=[question], return_tensors="pt"
            ).to(self.device)

            # get 'yes'
            yes_label = self.processor(text=["yes"], return_tensors="pt").to(self.device)

            with torch.no_grad():
                loss_yes = self.model(**inputs, labels=yes_label.input_ids).loss.item()  # type: ignore
            loss_list.append(loss_yes)
            # print(f'{property_}: {loss_yes}')

        return property_list[loss_list.index(min(loss_list))]

class VerifyPropertyInterpreter:
    def __init__(self, vqa_interpreter):
        self.vqa_interpreter = vqa_interpreter

    def __call__(self, image: Image.Image, object_name: str, property_: str) -> bool:
        is_plural = object_name.endswith("s")
        if is_plural:
            template = "Are the {object_name} {property}?"
        else:
            template = "Is the {object_name} {property}?"
        question = template.format(object_name=object_name, property=property_)
        inputs = self.vqa_interpreter.processor(
            images=[image], text=[question], return_tensors="pt"
        ).to(self.vqa_interpreter.device)

        # yes tokenizer
        yes_label = self.vqa_interpreter.processor(
            text=["yes"], return_tensors="pt"
        ).to(self.vqa_interpreter.device)
        # no tokenizer
        no_label = self.vqa_interpreter.processor(text=["no"], return_tensors="pt").to(
            self.vqa_interpreter.device
        )

        with torch.no_grad():
            loss_yes = self.vqa_interpreter.model(
                **inputs, labels=yes_label.input_ids
            ).loss.item()
            loss_no = self.vqa_interpreter.model(
                **inputs, labels=no_label.input_ids
            ).loss.item()

        return loss_yes < loss_no


class ExistsInterpreter:
    def __init__(self, vqa_interpreter):
        self.vqa_interpreter = vqa_interpreter

    def __call__(self, image: Image.Image, object_name: str) -> bool:
        question = "Is there a {}?".format(object_name)
        inputs = self.vqa_interpreter.processor(
            images=[image], text=[question], return_tensors="pt"
        ).to(self.vqa_interpreter.device)

        yes_label = self.vqa_interpreter.processor(
            text=["yes"], return_tensors="pt"
        ).to(self.vqa_interpreter.device)
        no_label = self.vqa_interpreter.processor(text=["no"], return_tensors="pt").to(
            self.vqa_interpreter.device
        )

        with torch.no_grad():
            loss_yes = self.vqa_interpreter.model(
                **inputs, labels=yes_label.input_ids
            ).loss.item()
            loss_no = self.vqa_interpreter.model(
                **inputs, labels=no_label.input_ids
            ).loss.item()

        return loss_yes < loss_no


class LlavaInterpreter:
    def __init__(self, device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_slug = "llava-hf/llava-1.5-7b-hf"
        #ours
        self.prompt_template = "<image>\nUSER: {user_message}\nASSISTANT:"
        
        #official
        self.prompt_template = "USER: <image>\n{user_message} ASSISTANT:"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_slug,
            low_cpu_mem_usage=True,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_slug,revision='a272c74')
        self.model.eval()  # type: ignore

    def predict(self, img: Image.Image, question: str, short_answer:bool=False) -> str:
        if short_answer: 
            question = f'{question}\nAnswer the question using a single word or phrase.'
        prompt = self.prompt_template.format(user_message=question)
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_length=200)  # type: ignore

        # Ignore the input tokens and only return the generated text.
        prompt_length = inputs.input_ids.shape[1]
        token_ids_to_decode = generate_ids[:, prompt_length:]
        generated_tokens = self.processor.batch_decode(
            token_ids_to_decode,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return generated_tokens

    def __call__(self, image: Image.Image, question: str) -> str:
        return self.predict(image, question, short_answer=True)
    
    def complex_query(self, image: Image.Image, question: str) -> str:
        return self.predict(image, question, short_answer=False)

    def verify_property(
        self, image: Image.Image, object_name: str, property_: str
    ) -> bool:
        return llava_verify_property(self, image, object_name, property_, template_last_pattern="ASSISTANT:")
    
    def best_description_from_options(
        self, image: Image.Image, object_name: str, property_list: list
    ) -> str:
        return llava_best_description_from_options(self, image, object_name, property_list, template_last_pattern="ASSISTANT:")
    
    def img_description_loss(
        self, image: Image.Image, object_name: str, property: str
    ) -> str:
        return img_description_loss(self, image, object_name, property, template_last_pattern="ASSISTANT:")

class Llava16Interpreter:
    def __init__(self, device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_slug = "llava-hf/llava-v1.6-mistral-7b-hf"
        self.prompt_template = "[INST] <image>\n {user_message} [/INST]"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_slug,
            low_cpu_mem_usage=True,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        
        self.processor = LlavaNextProcessor.from_pretrained(self.model_slug)
        self.model.eval()  # type: ignore

    def predict(self, img: Image.Image, question: str, short_answer:bool=False) -> str:
        if short_answer: 
            question = f'{question} Answer should be very concise and as short as possible.'
        prompt = self.prompt_template.format(user_message=question)
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_length=200)  # type: ignore
        # Ignore the input tokens and only return the generated text.
        prompt_length = inputs.input_ids.shape[1]

        token_ids_to_decode = generate_ids[:, prompt_length:]
        generated_tokens = self.processor.batch_decode(
            token_ids_to_decode,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return generated_tokens

    def __call__(self, image: Image.Image, question: str) -> str:
        return self.predict(image, question, short_answer=True)
    
    def verify_property(
        self, image: Image.Image, object_name: str, property_: str
    ) -> bool:
        return llava_verify_property(self, image, object_name, property_, template_last_pattern="[/INST]")
    def best_description_from_options(
        self, image: Image.Image, object_name: str, property_list: list
    ) -> str:
        return llava_best_description_from_options(self, image, object_name, property_list, template_last_pattern="[/INST]")
    
    def img_description_loss(
        self, image: Image.Image, object_name: str, property: str
    ) -> str:
        return img_description_loss(self, image, object_name, property, template_last_pattern="[/INST]")
    
class LlavaNextInterpreter:
    def __init__(self, device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_slug = "llava-hf/llama3-llava-next-8b-hf"
        self.prompt_template = "[INST] <image>\n {user_message} [/INST]"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_slug,
            low_cpu_mem_usage=True,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        
        self.processor = LlavaNextProcessor.from_pretrained(self.model_slug)
        self.model.eval()  # type: ignore

    def predict(self, img: Image.Image, question: str, short_answer:bool=False) -> str:

        if short_answer: 
            question = f'{question} Answer should be very concise and as short as possible.'

        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_length=200)  # type: ignore

        prompt_length = inputs.input_ids.shape[1]

        token_ids_to_decode = generate_ids[:, prompt_length:]
        prediction_output = self.processor.decode(token_ids_to_decode[0], skip_special_tokens=True)

        # output strip(), otherwise there is a '\n' before the answer
        return prediction_output.strip()

    def __call__(self, image: Image.Image, question: str) -> str:
        return self.predict(image, question, short_answer=True)
    
    def complex_query(self, image: Image.Image, question: str) -> str:
        return self.predict(image, question, short_answer=False)

    def verify_property(
        self, image: Image.Image, object_name: str, property_: str
    ) -> bool:
        return llava_verify_property(self, image, object_name, property_, template_last_pattern="[/INST]")
    
    def best_description_from_options(
        self, image: Image.Image, object_name: str, property_list: list
    ) -> str:
        return llava_best_description_from_options(self, image, object_name, property_list, template_last_pattern="[/INST]")
    
    def img_description_loss(
        self, image: Image.Image, object_name: str, property: str
    ) -> str:
        return img_description_loss(self, image, object_name, property, template_last_pattern="[/INST]")

class QwenInterpreter:
    def __init__(self, device="cuda:0", model_slug="Qwen/Qwen2-VL-7B-Instruct"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_slug = model_slug
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_slug,
            torch_dtype="auto",
            device_map="auto",
        )
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(self.model_slug, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        self.model.eval()  # type: ignore

    def predict(self, img: Image.Image, question: str, short_answer:bool=False) -> str:

        if short_answer: 
            question = f'{question} \nAnswer the question using a single word or phrase.'

        conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": question},
                                        ],
                        }
                    ]
        #get prompt and visual input
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info(conversation)

        inputs = self.processor(images=img, text=[prompt], padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=256)  # type: ignore

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
            ]
        prediction_output = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # output strip(), otherwise there is a '\n' before the answer
        return prediction_output[0]

    def __call__(self, image: Image.Image, question: str) -> str:
        return self.predict(image, question, short_answer=True)
    
    def complex_query(self, image: Image.Image, question: str) -> str:
        return self.predict(image, question, short_answer=False)

    def verify_property(
        self, image: Image.Image, object_name: str, property_: str
    ) -> bool:
        return llava_verify_property(self, image, object_name, property_, template_last_pattern="qwen")
    
    def best_description_from_options(
        self, image: Image.Image, object_name: str, property_list: list
    ) -> str:
        return llava_best_description_from_options(self, image, object_name, property_list, template_last_pattern="qwen")
    
    def img_description_loss(
        self, image: Image.Image, object_name: str, property: str
    ) -> str:
        return img_description_loss(self, image, object_name, property, template_last_pattern="qwen")

def llava_verify_property(self, image: Image.Image, object_name: str, property_: str, template_last_pattern:str) -> bool:
    is_plural = object_name.endswith("s")
    if is_plural:
        template = "Are the {object_name} {property}?"
    else:
        template = "Is the {object_name} {property}?"
    question = template.format(object_name=object_name, property=property_)

    if template_last_pattern != "qwen":
        # find out the last token inputid value.
        inst_end_last_token = self.processor.tokenizer(text=template_last_pattern, add_special_tokens=False).input_ids[-1]

        # tokenizing input+answer'yes' 
        prompt_yes = self.prompt_template.format(user_message=question) + ' yes'
        inputs_yes = self.processor(text=prompt_yes, images=image, return_tensors="pt").to(
            self.device
        )

        # tokenizing input+answer'no' 
        prompt_no = self.prompt_template.format(user_message=question) + ' no'
        inputs_no = self.processor(text=prompt_no, images=image, return_tensors="pt").to(
            self.device
        )

        yes_label = inputs_yes.input_ids.clone()

        # where to mask on answer 'yes'
        inputs_yes_until_loc = (yes_label == inst_end_last_token).nonzero(as_tuple=True)[1][1] +1 
        
        # put the mask on tensor execept 'yes' token.
        yes_label[:, :inputs_yes_until_loc] = -100

        no_label = inputs_no.input_ids.clone()
        # where to mask on answer 'no'
        inputs_no_until_loc = (no_label == inst_end_last_token).nonzero(as_tuple=True)[1][1] +1
        
        # put the mask on tensor execept 'no' token.
        no_label[:, :inputs_no_until_loc] = -100

        with torch.no_grad():
            loss_yes = self.model(**inputs_yes, labels=yes_label).loss.item()  # type: ignore
            loss_no = self.model(**inputs_no, labels=no_label).loss.item()  # type: ignore

    else:
        yes_no_loss = []
        for yes_no in ['yes','no']:
            conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": question},
                                            ],
                            }
                        ]
            # find out the last token inputid value.
            question_token_length = len(self.processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True))
            
            conversation.append({
                                    "role": "assistant",
                                    "content":[{"type": "text", "text": yes_no}]
            })

            #get prompt and visual input tokenize false, generation false
            prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

            inputs_ = self.processor(images=image, text=[prompt], padding=True, return_tensors="pt").to(self.device)

            #masking question part to avoid loss calculation
            inputs_id = inputs_.input_ids.clone()
            inputs_id[:, :question_token_length] = -100

            with torch.no_grad():
                loss_ = self.model(**inputs_, labels=inputs_id).loss.item()  # type: ignore
                yes_no_loss.append(loss_)
        loss_yes = yes_no_loss[0]
        loss_no = yes_no_loss[1]

    return loss_yes < loss_no    


def llava_best_description_from_options(self, image: Image.Image, object_name: str, property_list: list, template_last_pattern: str) -> str:
    is_plural = object_name.endswith("s")
    if is_plural:
        template = "Are the {object_name} {property}?"
    else:
        template = "Is the {object_name} {property}?"

    loss_list = []
    for property_ in property_list:
        question = template.format(object_name=object_name, property=property_)

        if template_last_pattern!= 'qwen':
            # find out the last token inputid value.
            inst_end_last_token = self.processor.tokenizer(text=template_last_pattern, add_special_tokens=False).input_ids[-1]

            # tokenizing input+answer'yes' 
            prompt_yes = self.prompt_template.format(user_message=question) + ' yes'
            inputs_yes = self.processor(text=prompt_yes, images=image, return_tensors="pt").to(
                self.device
            )


            yes_label = inputs_yes.input_ids.clone()

            # where to mask on answer 'yes'
            inputs_yes_until_loc = (yes_label == inst_end_last_token).nonzero(as_tuple=True)[1][1] +1 
            
            # put the mask on tensor execept 'yes' token.
            yes_label[:, :inputs_yes_until_loc] = -100

            with torch.no_grad():
                loss_yes = self.model(**inputs_yes, labels=yes_label).loss.item()  # type: ignore

            loss_list.append(loss_yes)
        else:
            conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": question},
                                            ],
                            }
                        ]
            # find out the last token inputid value.
            question_token_length = len(self.processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True))
            
            conversation.append({
                                    "role": "assistant",
                                    "content":[{"type": "text", "text": property_}]
            })

            #get prompt and visual input tokenize false, generation false
            prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

            inputs_ = self.processor(images=image, text=[prompt], padding=True, return_tensors="pt").to(self.device)

            #masking question part to avoid loss calculation
            inputs_id = inputs_.input_ids.clone()
            inputs_id[:, :question_token_length] = -100

            with torch.no_grad():
                loss_ = self.model(**inputs_, labels=inputs_id).loss.item()  # type: ignore
                loss_list.append(loss_)

    return property_list[loss_list.index(min(loss_list))]

def img_description_loss(self, image: Image.Image, object_name: str, property_: str, template_last_pattern: str):
    is_plural = object_name.endswith("s")
    if is_plural:
        template = "Are the {object_name} {property}?"
    else:
        template = "Is the {object_name} {property}?"

    # property_ = [property]
    loss_list = []
    # for property_ in property_list:
    question = template.format(object_name=object_name, property=property_)

    if template_last_pattern!= 'qwen':
        # find out the last token inputid value.
        inst_end_last_token = self.processor.tokenizer(text=template_last_pattern, add_special_tokens=False).input_ids[-1]

        # tokenizing input+answer'yes' 
        prompt_yes = self.prompt_template.format(user_message=question) + ' yes'
        inputs_yes = self.processor(text=prompt_yes, images=image, return_tensors="pt").to(
            self.device
        )


        yes_label = inputs_yes.input_ids.clone()

        # where to mask on answer 'yes'
        inputs_yes_until_loc = (yes_label == inst_end_last_token).nonzero(as_tuple=True)[1][1] +1 
        
        # put the mask on tensor execept 'yes' token.
        yes_label[:, :inputs_yes_until_loc] = -100

        with torch.no_grad():
            loss_yes = self.model(**inputs_yes, labels=yes_label).loss.item()  # type: ignore

        loss_list.append(loss_yes)
    else:
        conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": question},
                                        ],
                        }
                    ]
        # find out the last token inputid value.
        question_token_length = len(self.processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True))
        
        conversation.append({
                                "role": "assistant",
                                "content":[{"type": "text", "text": property_}]
        })

        #get prompt and visual input tokenize false, generation false
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

        inputs_ = self.processor(images=image, text=[prompt], padding=True, return_tensors="pt").to(self.device)

        #masking question part to avoid loss calculation
        inputs_id = inputs_.input_ids.clone()
        inputs_id[:, :question_token_length] = -100

        with torch.no_grad():
            loss_ = self.model(**inputs_, labels=inputs_id).loss.item()  # type: ignore
            loss_list.append(loss_)

    return loss_list[0]


    
