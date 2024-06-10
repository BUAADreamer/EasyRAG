#!/usr/bin/env python3

import json
import argparse
from urllib import request, error

# 提交答案服务域名或IP
JUDGE_SERVER = "http://judge.aiops-challenge.com"
# 比赛ID，字符串类型，可通过比赛界面 URL 获取, 比如"赛道一（Qwen1.5-14B）：基于检索增强的运维知识问答挑战赛"的URL为https://competition.aiops-challenge.com/home/competition/1771009908746010681 ，比赛ID为1771009908746010681
CONTEST = "1780211530478944282"
# 团队ID, 字符串类型，需要在参加比赛并组队后能获得，具体在比赛详情页-> 团队 -> 团队ID，为一串数字标识。 
TICKET = "1794384620607799320"


def submit(data, judge_server=None, contest=None, ticket=None):
    judge_server = judge_server or JUDGE_SERVER
    contest = contest or CONTEST
    ticket = ticket or TICKET

    if not judge_server or not contest or not ticket:
        missing = [
            "judge_server" if not judge_server else "",
            "contest" if not contest else "",
            "ticket" if not ticket else "",
        ]
        missing = [m for m in missing if m]
        print("Required fields must be provided: %s" % ', '.join(missing))
        return None

    req_data = json.dumps({'data': data}).encode('utf-8')
    req = request.Request(judge_server, data=req_data, headers={'ticket': ticket, 'contest': contest, 'Content-Type': 'application/json'})

    try:
        with request.urlopen(req) as response:
            response_body = response.read().decode('utf-8')
            return json.loads(response_body)['submission_id']
    except error.HTTPError as e:
        msg = e.reason
        response_body = e.read().decode('utf-8')
        if response_body:
            try:
                msg = json.loads(response_body)['detail']
            except:
                pass
        print("[Error %s] %s" % (e.code, msg))

    except error.URLError as e:
        print(e.reason)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit to judge server")
    parser.add_argument('result_path', nargs='?', default='result.jsonl', help='Path to the submission file, default is result.jsonl')
    parser.add_argument('-s', '--server', help='Judge server URL, if not specified, the global JUDGE_SERVER variable will be used')
    parser.add_argument('-c', '--contest', help='Contest ID, if not specified, the global CONTEST variable will be used')
    parser.add_argument('-k', '--ticket', help='Submission ticket, if not specified, the global TICKET variable will be used')

    args = parser.parse_args()

    try:
        with open(args.result_path, 'r') as file:
            data = [json.loads(line.strip()) for line in file if line.strip()]
    except Exception as e:
        print(e)
        exit(1)

    submission_id = submit(data, judge_server=args.server, contest=args.contest, ticket=args.ticket)
    if submission_id:
        print("Success! Your submission ID is %s." % submission_id)
        exit(0)
    else:
        exit(1)
