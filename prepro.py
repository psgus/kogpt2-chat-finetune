from pathlib import Path
import json
import re

SINGLE_REMOVALS = {
    "키", "하", "덜", "후", "껄", "쩝", "오", "아", "헤", "흠", "허",
    "음", "히", "우", "호", "헐", "흐", "풉", "크", "휴", "읏", "웃",
    "얏", "윽", "흣", "얍", "흑", "읍"
}
REMOVE_WORDS = [
    "에효", "어허", "아이구", "아이쿠", "에이구", "아하하", "오호",
    "헤헷", "ㅎㅋ", "음흠", "흐음", "하하핳", "어휴", "힣", "킼", "[이모티콘]",
    "하히핳", "야호", "에헤", "에이", "아야", "어머", "어우", "에휴", "호이", "호잇", "아하", "어라",
    "으흠", "에구", "이야", "이익", "옹옹", "읏차", "으아",
    "으앙", "으흣", "으응", "에잇", "에헷", "에헤라", "어잉", "에이구",
    "우와", "으휴", "으흐", "으쓱", "에구구", "어이쿠", "어흠", "어헣", "우웩", "으익", "아구", "와우", "어흑", "야옹", "오잉", "얏호", "으이그", "음야"
]
REMOVE_WORDS_SET = set(REMOVE_WORDS)
TARGET_FOLDERS = [
    Path(r"020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/원천데이터/TS_01. KAKAO(1)"),
    Path(r"020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/원천데이터/TS_01. KAKAO(2)"),
    Path(r"020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/원천데이터/TS_01. KAKAO(3)"),
    Path(r"020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/원천데이터/TS_01. KAKAO(4)"),
    Path(r"020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/원천데이터/TS_02. FACEBOOK"),
    Path(r"020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/원천데이터/TS_03. INSTAGRAM"),
    Path(r"020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/원천데이터/TS_04. BAND"),
    Path(r"020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/원천데이터/TS_05. NATEON"),
]
output_path = Path("111.jsonl")


def clean_special_char(s: str) -> str:
    s = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", s)
    s = s.lower()
    s = re.sub(r'([a-z]+)', r' \1 ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    for word in SINGLE_REMOVALS:
        pattern_repeat = re.compile(rf'(?:{re.escape(word)}){{2,}}')
        s = pattern_repeat.sub('', s)

    tokens = s.strip().split()
    filtered_tokens = []
    for tok in tokens:
        if tok in SINGLE_REMOVALS:
            continue
        single_pattern = re.compile(r'^(' + '|'.join(re.escape(w) for w in SINGLE_REMOVALS) + r'){2,}$')
        if single_pattern.match(tok):
            continue
        if tok in REMOVE_WORDS_SET:
            continue
        filtered_tokens.append(tok)

    return ' '.join(filtered_tokens).strip()


def merge_same_speaker(lines: list) -> list:
    merged_lines = []
    prev_speaker = None
    buffer = []

    for line in lines:
        if ':' not in line:
            continue
        speaker, utterance = line.split(":", 1)
        speaker = speaker.strip()
        utterance = utterance.strip()
        if prev_speaker is None:
            buffer = [utterance]
            prev_speaker = speaker
        elif speaker == prev_speaker:
            buffer.append(utterance)
        else:
            merged_lines.append(" ".join(buffer))
            buffer = [utterance]
            prev_speaker = speaker

    if buffer:
        merged_lines.append(" ".join(buffer))
    return merged_lines


if __name__ == "__main__":
    file_count = 0
    pair_count = 0
    with open(output_path, "w", encoding="utf-8") as out_file:
        for folder in TARGET_FOLDERS:
            for file in folder.glob("*.txt"):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        file_count += 1
                        raw_lines = [line.strip() for line in f if line.strip()]
                        merged = merge_same_speaker(raw_lines)
                        cleaned = [clean_special_char(ln) for ln in merged]
                        if any('*' in ln for ln in cleaned):
                            continue
                        for i in range(len(cleaned) - 1):
                            inp = cleaned[i]
                            out = cleaned[i + 1]
                            if 5 <= len(inp) <= 150 and 5 <= len(out) <= 150:
                                data = {"input": inp, "output": out}
                                out_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                                pair_count += 1
                except Exception as e:
                    print(f"파일 오류: {file} - {e}")

    print(f"✅ 정제 완료! jsonl 저장 위치: {output_path}")
    print(f"총 읽은 텍스트 파일 개수: {file_count}")
    print(f"총 저장된 jsonl 페어 개수: {pair_count}")
