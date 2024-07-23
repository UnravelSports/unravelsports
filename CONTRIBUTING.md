Contributing to 🌀 **unravelsports**
-----

Below is an outline of the steps you can follow if you want to contribute.

#### 1. Create a [**fork**](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of the ***unravelsports*** repository 
#### 2. Create a ***new branch*** on your fork with an easily identifiable name, for example:
```bash
git checkout -b feat/some-new-feature
```
or 
```bash
git checkout -b bug/fix-some-bug
```
#### 3. Make a fix or addition to your new branch.
#### 4. If applicable add pytest test(s) to the `tests/` directory and verify they are successful by running:
```bash
pytest tests/test_my_new_test.py
```
#### 5. Ensure the code conforms to the coding standards by running:
```bash
black .
```
Make sure you have **black** installed by running:
```bash
pip install bash[jupyter]
```
#### 6. Commit and push the changes to your fork
```bash
git add the_file_you_changed.py
git commit -m "I fixed a thing"
git push --set-upstream origin bug/fix-some-bug
```
#### 7. Navigate to [***unravelsports*** Pull requests](https://github.com/UnravelSports/unravelsports/pulls) and create ***New pull request*** that merges your `bug/fix-some-bug` branch into `main`

Project Setup
-----

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```