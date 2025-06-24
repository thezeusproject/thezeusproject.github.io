# Zeus Project

Welcome to the Zeus Project! This repository is a collection of educational posts and associated projects related to various topics in machine learning and data science. We aim to provide clear theoretical explanations along with hands-on projects to solidify understanding.

We welcome contributions from the community! If you have a topic you'd like to write about, please follow the steps below.

## How to Contribute

To contribute a new blog post and project, you'll need to add two files: one for the blog content and one for the project description.

### Step 1: Fork and Clone the Repository

Start by forking the repository to your own GitHub account and then clone it to your local machine.

```bash
git clone https://github.com/<your-username>/thezeusproject.github.io
cd thezeusproject.github.io
```
> Replace `<your-username>` with your GitHub username

### Step 2: Create New Content Files

A complete contribution consists of a blog post, a corresponding project, and an HTML page to display them.

1.  **Create a blog post**: Navigate to the `content/blog` directory. The easiest way to start is to copy an existing blog post file, for example `linear-regression.md`, and rename it to your new topic (e.g., `your-topic-name.md`).

2.  **Create a project file**: Similarly, go to `content/projects` and create a corresponding markdown file for the project tasks (e.g., `your-topic-name.md`). You can also copy an existing file from this directory to use as a template.

3.  **Create an HTML page**: Go to `pages/blog-pages`. Copy an existing HTML file (e.g., `linear-regression.html`) and rename it to match your topic (e.g., `your-topic-name.html`).

### Step 3: Add Your Content

Now, open the files you've created and add your content.

#### Blog Post (`content/blog/your-topic-name.md`)

At the beginning of your blog post file, you need to include a frontmatter section. This is crucial for the website to correctly display your post.

Here is an example of what the frontmatter should look like:

```markdown
<!-- filler line (always add this filler line or code wont work)-->
---
title: Your Blog Post Title
subtitle: A brief and engaging subtitle for your post.
author: your-github-username
---

![image](assets/images/your-awesome-image.png)

## Section 1: Introduction

Your amazing content starts here...
```

-   **`title`**: The title of your blog post.
-   **`subtitle`**: A short description that will appear under the title.
-   **`author`**: Your name or username.
-   **Image**: You can add a header image. **All images used in your blog post or project must be uploaded to the `/assets/images/` directory.** Link to them as shown above (e.g., `![image](assets/images/your-awesome-image.png)`).

After the frontmatter, you can write the full content of your blog post in Markdown.

#### Project File (`content/projects/your-topic-name.md`)

The project file should contain a list of tasks for a hands-on project related to your blog post. Use Markdown checklists to create the tasks.

Here is an example structure:

```markdown
### Project: Your Project Title

- [ ] **Task 1: A description of the first task**
  - [ ] A sub-task if needed.
- [ ] **Task 2: Another task**
```

### Step 4: Update the HTML file

Open the new HTML file you created (`pages/blog-pages/your-topic-name.html`) and make sure the title and the paths to the markdown files are correct.

- **Update Title**: Change the `<title>` tag to reflect your blog post's title.
- **Update Content Paths**: Locate the `fetch` calls in the JavaScript section at the bottom of the file. Update the paths to point to your new markdown files in `content/blog` and `content/projects`.

For example, if your topic is `your-topic-name`, the `fetch` calls should look like this:

```javascript
// ...
Promise.all([
    fetch('../../content/blog/your-topic-name.md').then(res => res.text()),
    fetch('../../content/projects/your-topic-name.md').then(res => res.text())
])
//...
```

### Step 5: Submit a Pull Request

Once you have created your blog post and project files, commit the changes and push them to your fork.

```bash
git add .
git commit -m "feat: Add new blog on [Your Topic]"
git push origin main
```

Then, go to the original repository on GitHub and open a Pull Request. We will review your contribution and merge it if everything looks good.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Thank you for contributing to the Zeus Project! 
