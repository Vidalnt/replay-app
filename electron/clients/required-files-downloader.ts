import type { AxiosProgressEvent } from "axios";
import axios from "axios";
import axiosRetry from "axios-retry";
import logger from "../../shared/logger";
import fs from "fs";
import fsp from "fs/promises";
import { localWeightsPath } from "../utils/constants";
import path from "path";
import jetpack from "fs-jetpack";
import { httpAgent, httpsAgent } from "./agents.ts";

axiosRetry(axios, { retries: 2 });

const HF_BASE_URL = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources";

type WeightFile = {
  name: string;
  remoteFolder: string;
  localSubPath: string;
  size: number;
  sha1: string;
};

const INFERENCE_FILES: WeightFile[] = [
  {
    name: "rmvpe.pt",
    remoteFolder: "predictors/",
    localSubPath: "predictors/",
    size: 181184272,
    sha1: "a620a75526053cf06aa43bb2d50c29ad64cc0d9b",
  },
  {
    name: "fcpe.pt",
    remoteFolder: "predictors/",
    localSubPath: "predictors/",
    size: 43362881,
    sha1: "fea8704a8f57523eb484435b99793cac4f6921e5",
  },
  {
    name: "pytorch_model.bin",
    remoteFolder: "embedders/contentvec/",
    localSubPath: "embedders/contentvec/",
    size: 378342945,
    sha1: "f3db65dcdb1af732bc48c8e1ba5e17d4f84d5eb8",
  },
  {
    name: "config.json",
    remoteFolder: "embedders/contentvec/",
    localSubPath: "embedders/contentvec/",
    size: 1388,
    sha1: "fed1d97aeaf83ef91f4c9a33c31ed64a5ddf29fd",
  },
];

class RequiredFilesController {
  progress: null | AxiosProgressEvent = null;
  isDownloading: boolean = false;
  error: null | object = null;
  fileCount: number = INFERENCE_FILES.length;
  totalSize: number = INFERENCE_FILES.reduce((acc, f) => acc + f.size, 0);
  totalSizeDownloaded: number = 0;
  currentFileNum: number = 0;

  private getFullPath = (file: WeightFile) => path.join(localWeightsPath, file.localSubPath, file.name);
  private getUrl = (file: WeightFile) => `${HF_BASE_URL}/${file.remoteFolder}${file.name}`;

  getLocallyDownloadedModelFiles = async () => {
    const files: string[] = [];
    for (const file of INFERENCE_FILES) {
      if (await jetpack.existsAsync(this.getFullPath(file))) {
        files.push(file.name);
      }
    }
    return files;
  };

  hasDownloadedModelWeights = async () => {
    for (const file of INFERENCE_FILES) {
      const stat = await fsp.stat(this.getFullPath(file)).catch(() => null);
      if (!stat || stat.size !== file.size) return false;
    }
    return true;
  };

  removeModelWeights = async () => {
    await Promise.all(
      INFERENCE_FILES.map((file) => jetpack.removeAsync(this.getFullPath(file)))
    );
  };

  verifyModelWeights = async () => {
    for (const file of INFERENCE_FILES) {
      const stat = await jetpack.inspectAsync(this.getFullPath(file), { checksum: "sha1" }).catch(() => null);
      if (!stat?.sha1 || stat.sha1 !== file.sha1) return false;
    }
    return true;
  };

  listAndDownloadModels = async () => {
    if (this.isDownloading) return;
    
    this.progress = null;
    this.isDownloading = true;
    this.error = null;
    this.totalSizeDownloaded = 0;
    this.currentFileNum = 0;

    try {
      for (const file of INFERENCE_FILES) {
        this.currentFileNum++;
        this.progress = null;
        await this.download(file);
        this.totalSizeDownloaded += file.size;
      }
    } catch (e) {
      this.error = e;
      logger.error(e);
      throw e;
    } finally {
      this.isDownloading = false;
    }
  };

  private download = async (file: WeightFile) => {
    const fullPath = this.getFullPath(file);
    const url = this.getUrl(file);

    const stat = await jetpack.inspectAsync(fullPath, { checksum: "sha1" }).catch(() => null);
    if (stat?.size === file.size && stat?.sha1 === file.sha1) {
      return;
    }

    await jetpack.dirAsync(path.dirname(fullPath));

    const response = await axios.get(url, {
      responseType: "stream",
      onDownloadProgress: (progressEvent) => {
        this.progress = progressEvent;
      },
      httpsAgent,
      httpAgent,
      timeout: 30000,
    });

    const writer = fs.createWriteStream(fullPath, { flags: "w" });
    response.data.pipe(writer);

    await new Promise<void>((resolve, reject) => {
      writer.on("finish", async () => {
        const downloaded = await jetpack.inspectAsync(fullPath, { checksum: "sha1" });
        if (downloaded?.sha1 !== file.sha1) {
          await jetpack.removeAsync(fullPath);
          reject(new Error(`SHA1 verification failed for ${file.name}`));
          return;
        }
        resolve();
      });
      
      writer.on("error", (error) => {
        logger.error(error);
        this.error = error;
        reject(error);
      });
      
      response.data.on("error", (error: any) => {
        logger.error(error);
        this.error = error;
        reject(error);
      });
    });
  };
}

export const requiredFilesController = new RequiredFilesController();