import pandas as pd
from src.prompters import RetrieveInContextExamplesByQtypePrompter
from src.instrumentation import calculate_vqa_v2_exact_match_score, calculate_macro_f1_and_mean_iou_from_evaluation_records
import yaml
import json
import os
import argparse
import copy
from src.dataset_io import VqaDatasetWithImageRoot

def rreplace(s, old, new):
    li = s.rsplit(old, 1) #Split only once
    return new.join(li)

def main(args):
    print(args.instruction_mode)
    if args.instruction_deeper:
        args.instruction_mode = True

    # Initialize and compose the configuration
    template_file_path = os.path.join('prompts', args.template_file_path)

    # Read the file as a string
    with open(template_file_path, 'r', encoding='utf-8') as file:
        prompte_template = file.read()

    ice_file_path = os.path.join('in_context_examples', args.ice_file_path) 
    with open(ice_file_path, 'r', encoding='utf-8') as file:
        ice = yaml.safe_load(file)

    # raw data
    raw_dataset_path = args.raw_dataset_path 
    raw_data_records = []
    with open(raw_dataset_path, 'r', encoding='utf-8') as file:
        for line in file:
            raw_data_records.append(json.loads(line))

    #prompter
    if 'okvqa' in raw_dataset_path:
        raw_data_records = VqaDatasetWithImageRoot(args.okvqa_image_root, raw_dataset_path)
    if 'refcoco' in raw_dataset_path:
        raw_data_records = VqaDatasetWithImageRoot(args.refcoco_image_root, raw_dataset_path)

    prompter = RetrieveInContextExamplesByQtypePrompter(prompte_template, ice, raw_data_records)

    # output file and folder
    output_file_path = os.path.join(args.cache_parent_folder_name, args.output_file_folder_name, 'output_file.jsonl') 
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # inference records_file
    inference_records_file_path = os.path.join(args.cache_parent_folder_name, args.inference_records_folder_name, 'records.jsonl')
    inference_records_df = pd.read_json(inference_records_file_path, lines=True)

    def parsing_obs_renderer(obs_, instruction_mode=False, step_no=1):
        if instruction_mode:
            template = f"<result>REPLACING_THIS_PART</result>\nStep {step_no}:"
        else:
            template = "<result>REPLACING_THIS_PART</result>"
        if obs_["observation_type"] =="code_observation":
            return template.replace('REPLACING_THIS_PART', obs_['execution_result'])
        elif obs_["observation_type"] =="non_code_observation":
            return template.replace('REPLACING_THIS_PART', obs_['content'])
        elif obs_["observation_type"] =="null_observation":
            return template.replace('REPLACING_THIS_PART', '')
        return None


    whole_dataset_for_training = []
    for index_, row_ in inference_records_df.iterrows():
        
        if 'refcoco' in raw_dataset_path:
            append_flag = calculate_macro_f1_and_mean_iou_from_evaluation_records([row_])["mean_iou"]>0.8
        else:
            append_flag = calculate_vqa_v2_exact_match_score(row_["result"],row_["label"])

        # only collect correct trajectory
        if append_flag:
            # print('yes')
            one_question_prompt = prompter(row_["question"], row_["caption"]) #row_["caption"]

            if args.instruction_deeper:
                step_no = 2
                #calculate trajectory number
                sub_num_trajectory = len(row_.trajectory)

                # only for trajectory >3
                if sub_num_trajectory > 3:
                    # from 2 to n-1
                    for short_step_num in range(2, sub_num_trajectory):
                        step_no = 2
                        one_question_prompt_copy = copy.deepcopy(one_question_prompt)
                        for one_trajectory in row_.trajectory:
                            # get the whole conversation
                            row_action = one_trajectory["action"]
                            row_obs = parsing_obs_renderer(one_trajectory["observation"], args.instruction_mode, step_no)
                            one_question_prompt_copy = '\n'.join([one_question_prompt_copy,row_action,row_obs])
                            
                            # if longer than short_step_num stop
                            if step_no > short_step_num:
                                break
                            step_no+=1
                    
                        # then do the mask on range(0, short_step_num-1)
                        for step_no in range(short_step_num-1):
                            # print(f'short_step_num:{short_step_num}')
                            # print(f'step_no{step_no}')
                            # return
                            one_datapoint = []
                            # user info
                            ideal_answer = row_.trajectory[step_no]["action"]
                            sub_question_prompt = rreplace(one_question_prompt_copy, ideal_answer, '[You need to fill this part]')
                            #remove last step obs (meaningless) and write down instruction step no:
                            sub_question_prompt = rreplace(sub_question_prompt, row_obs, f'Your solution to Step {step_no+1}:')
                            one_datapoint.append({"content": sub_question_prompt,"role":"user"})

                            #system assistant
                            one_datapoint.append({"content": ideal_answer,"role":"assistant"})

                            # save as datapoint
                            whole_dataset_for_training.append(one_datapoint)


            one_question_prompt_copy = copy.deepcopy(one_question_prompt)
            if args.instruction_mode:
                step_no = 2
                for one_trajectory in row_.trajectory:
                    # get the whole conversation
                    row_action = one_trajectory["action"]
                    row_obs = parsing_obs_renderer(one_trajectory["observation"], args.instruction_mode, step_no)
                    one_question_prompt_copy = '\n'.join([one_question_prompt_copy,row_action,row_obs])
                    step_no+=1

                # one_question_prompt_copy = copy.deepcopy(one_question_prompt)
                # masking
                for step_no in range(len(row_.trajectory)):
                    
                    # skip masking error
                    if "Traceback" in parsing_obs_renderer(row_.trajectory[step_no]["observation"], args.instruction_mode, step_no) or ("You are not allowed to use imports" in parsing_obs_renderer(row_.trajectory[step_no]["observation"], args.instruction_mode, step_no)):
                        continue
                        
                    one_datapoint = []
                    # user info
                    ideal_answer = row_.trajectory[step_no]["action"]
                    sub_question_prompt = rreplace(one_question_prompt_copy, ideal_answer, '[You need to fill this part]')
                    #remove last step obs (meaningless) and write down instruction step no:
                    sub_question_prompt = rreplace(sub_question_prompt, row_obs, f'Your solution to Step {step_no+1}:')
                    one_datapoint.append({"content": sub_question_prompt,"role":"user"})

                    #system assistant
                    one_datapoint.append({"content": ideal_answer,"role":"assistant"})

                    # save as datapoint
                    whole_dataset_for_training.append(one_datapoint)
                    

            # else:
            one_datapoint = []
            user_info_container = ""
            # combing_flag = False
            user_info_container += one_question_prompt
            
            # init - normal
            if "Traceback" in parsing_obs_renderer(row_.trajectory[0]["observation"], args.instruction_mode, 2) or ("You are not allowed to use imports" in parsing_obs_renderer(row_.trajectory[0]["observation"], args.instruction_mode, 2)):
                combing_flag = True
            else:
                combing_flag = False
                one_iteration = {"content": user_info_container,"role":"user"}
                one_datapoint.append(one_iteration)

            # step_no = 2
            for step_no in range(len(row_.trajectory)):
                    
                    # current step
                    one_trajectory = row_.trajectory[step_no]

                    # current obs
                    observation_parsing = parsing_obs_renderer(one_trajectory["observation"], args.instruction_mode, step_no+2)

                    # if combing_flag is true which mean this turn should skip and direct go to next turn.
                    if combing_flag:
                        user_info_container  = user_info_container + one_trajectory["action"] + observation_parsing
                    else:
                        one_iteration = {"content": one_trajectory["action"],"role":"assistant"}
                        one_datapoint.append(one_iteration)
                        user_info_container = observation_parsing

                    # if error in next turn, we would skip
                    if step_no+1 < len(row_.trajectory) and ("Traceback" in parsing_obs_renderer(row_.trajectory[step_no+1]["observation"], args.instruction_mode, step_no+3) or "You are not allowed to use imports" in parsing_obs_renderer(row_.trajectory[step_no+1]["observation"], args.instruction_mode, step_no+3)):
                        combing_flag = True
                    
                    else:
                        combing_flag=False
                        one_iteration = {"content": user_info_container,"role":"user"}
                        one_datapoint.append(one_iteration)


            whole_dataset_for_training.append(one_datapoint[:-1])
        # break

    #save training data file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for entry in whole_dataset_for_training:
            file.write(json.dumps(entry) + '\n')
    print('done')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--template_file_path", type=str)
    parser.add_argument("--ice_file_path", type=str)
    
    # raw_data_source
    parser.add_argument("--raw_dataset_path", type=str)
    parser.add_argument("--okvqa_image_root", type=str, default="")
    parser.add_argument("--refcoco_image_root", type=str, default="")
    
    # output_data_path 
    parser.add_argument("--output_file_folder_name", type=str)

    # inference_trajectory_path
    parser.add_argument("--inference_records_folder_name", type=str)

    parser.add_argument("--cache_parent_folder_name", type=str)

        
    # interaction mode
    parser.add_argument("--instruction_mode", action="store_true")
    parser.add_argument("--instruction_deeper", action="store_true")

    args = parser.parse_args()
    print(args.instruction_mode)
    print(args.cache_parent_folder_name)
    main(args)
