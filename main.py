"""실행 입구. 처리 순서는 object_tracker/app.py의 run()에서 읽습니다."""

from object_tracker.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
