const os = require("node:os");
const path = require("node:path");

const DEFAULT_DIRECTORY = path.resolve(os.homedir(), "AppData/Local/nomic.ai/gpt4all");

const librarySearchPaths = [
    path.join(DEFAULT_DIRECTORY, "libraries"),
    path.resolve("./libraries"),
    path.resolve(
        __dirname,
        "..",
        `runtimes/${process.platform}-${process.arch}/native`,
    ),
    //for darwin. This is hardcoded for now but it should work
    path.resolve(
        __dirname,
        "..",
        `runtimes/${process.platform}/native`,
    ),
    process.cwd(),
];

const DEFAULT_LIBRARIES_DIRECTORY = librarySearchPaths.join(";");

const DEFAULT_MODEL_CONFIG = {
    systemPrompt: "",
    promptTemplate: "### Scott:\n%1\n\n### Rommie:\n%2",
}

const DEFAULT_MODEL_LIST_URL = "c:/romi/models3.json";

const DEFAULT_PROMPT_CONTEXT = {
    temp: 1.5,
    topK: 40,
    topP: 0.4,
    minP: 0.0,
    repeatPenalty: 1.18,
    repeatLastN: 128,
    nBatch: 96,
}

module.exports = {
    DEFAULT_DIRECTORY,
    DEFAULT_LIBRARIES_DIRECTORY,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_MODEL_LIST_URL,
    DEFAULT_PROMPT_CONTEXT,
};
