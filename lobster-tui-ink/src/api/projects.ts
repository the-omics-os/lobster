/** Projects & datasets REST API (protocol §4.5). */

import type { AppConfig } from "../config.js";
import { apiFetch } from "./apiClient.js";
import { freshAuthHeaders } from "../config.js";

export interface Project {
  id: string;
  name: string;
  description?: string;
  dataset_count?: number;
  created_at?: string;
}

export interface Dataset {
  id: string;
  name: string;
  project_id: string;
  file_count?: number;
  size_bytes?: number;
  created_at?: string;
}

interface ProjectsResponse {
  projects: Project[];
}

interface DatasetFilesResponse {
  files: Array<{ name: string; size_bytes: number; url: string }>;
}

/** List all projects for the current user. */
export async function listProjects(config: AppConfig): Promise<Project[]> {
  try {
    const data = await apiFetch<ProjectsResponse>(config, "/projects");
    return data.projects ?? [];
  } catch {
    return [];
  }
}

/** Upload a file to a dataset. */
export async function pushFile(
  config: AppConfig,
  datasetId: string,
  filePath: string,
  fileContent: ArrayBuffer,
): Promise<string> {
  const url = `${config.apiUrl}/datasets/${datasetId}/files`;
  const formData = new FormData();
  formData.append("file", new Blob([fileContent]), filePath);

  const auth = await freshAuthHeaders(config);
  const resp = await fetch(url, {
    method: "POST",
    headers: auth,
    body: formData,
  });

  if (!resp.ok) {
    throw new Error(`Upload failed: ${resp.status} ${resp.statusText}`);
  }

  const data = (await resp.json()) as { file_id?: string; message?: string };
  return data.message ?? data.file_id ?? "Uploaded successfully";
}

/** List files in a dataset (for pull). */
export async function listDatasetFiles(
  config: AppConfig,
  datasetId: string,
): Promise<DatasetFilesResponse["files"]> {
  try {
    const data = await apiFetch<DatasetFilesResponse>(
      config,
      `/datasets/${datasetId}/files`,
    );
    return data.files ?? [];
  } catch {
    return [];
  }
}
