import os
import shutil
from datetime import datetime

chroma_base = "./chroma_db"

if not os.path.exists(chroma_base):
    print("No databases found.")
    exit(0)

dbs = []
for name in sorted(os.listdir(chroma_base)):
    path = os.path.join(chroma_base, name)
    if not os.path.isdir(path):
        continue
    size_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(path)
        for f in files
    )
    size_mb = size_bytes / (1024 * 1024)
    mtime = os.path.getmtime(path)
    date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    dbs.append((name, size_mb, date, path))

if not dbs:
    print("No databases found.")
    exit(0)

print("\nStored databases:")
for i, (name, size, date, _) in enumerate(dbs, 1):
    print(f"  {i}. {name:<40} ({size:.1f}MB, {date})")

total_mb = sum(s for _, s, _, _ in dbs)
print(f"\nTotal: {len(dbs)} databases, {total_mb:.1f}MB\n")

target = input("Enter number to delete (all: delete all, q: cancel): ").strip().lower()

if target == "q":
    print("Cancelled.")
elif target == "all":
    confirm = input(f"Delete all {len(dbs)} databases? (y/n): ").strip().lower()
    if confirm == "y":
        shutil.rmtree(chroma_base)
        print("All databases deleted.")
    else:
        print("Cancelled.")
elif target.isdigit() and 1 <= int(target) <= len(dbs):
    name, size, _, path = dbs[int(target) - 1]
    confirm = input(f"Delete '{name}' ({size:.1f}MB)? (y/n): ").strip().lower()
    if confirm == "y":
        shutil.rmtree(path)
        print(f"Deleted: {name}")
    else:
        print("Cancelled.")
else:
    print("Invalid input. Please enter a valid number.")
