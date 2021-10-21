- Use `-h` for help for any git command

### step1: Clone (download) the repo

```
$ git clone <link-to-repo>
```

eg. `git clone https://github.com/genesys-ai/jaipur-asset-detection`
you will see dir created in same location  where you ran the above command

### step2: Know what branch you are in
```
$ git branch -l
```

### Step 3.1  Push (upload) into a main branch

- First run, to update code

```
git pull origin <branch-name>
```
eg. `git pull origin main`

- Second, make changes to your code in the repo folder
- Third, run the below commands
  ```
  $ git add  .
  $ git commit -m"your message what you have changed"
  $ git push origin <branch name>
  ```
  eg. 
  ```
  $ git add  .
  $ git commit -m"added code to filter only specific classes"
  $ git push origin main
  ```

### Step 3.2  Push (upload) into a specific branch

- create **NEW** branch and get into the new branch using

```
$ git checkout -b <new-branch-name>
```

eg, `git checkout -b research` and to know if it was created `git branch`

- get into already existing branch

```
$ git checkout <existing-branch-name>
```

- make changes to code
- push into that specific branch

```
  $ git add  .
  $ git commit -m"your message what you have changed"
  $ git push origin <branch name>
```
eg.
```
  $ git add  .
  $ git commit -m"modified file"
  $ git push origin research
```

### Step4: Send pull request (to get changes from one branch into another)

- goto that branch using github website
- click on `pull request`
- change `to` <- `from` in the top bar
- add comments
- add reviewer if you want to
- click on `merge pr`

### Step5: Remove file 
```
git rm file1.txt
git commit -m "remove file1.txt"
git push origin <branch_name>
```

