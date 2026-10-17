import { useEffect, useMemo, useState } from "react";
import classNames from "classnames";

import { useAnnotationStore, labelActionKey, facescapeLabelActionKey } from "@/state/useAnnotationStore";
import { useAva256Store } from "@/state/useAva256Store";
import { LABEL_STATUSES, type LabelStatus } from "@/lib/types";

// Display names only -- deliberately different from the underlying LabelStatus
// values (which stay as-is in label_tracker.json/the backend API, see
// lib/types.ts), since the raw status names don't read clearly as a pipeline
// progression on their own.
const LABEL_STATUS_META: Record<LabelStatus, { text: string; badgeClass: string; pillActiveClass: string }> = {
  no_neutral_label: {
    text: "Unlabeled Neutral",
    badgeClass: "text-slate-500",
    pillActiveClass: "border-slate-500 text-slate-300 bg-slate-800"
  },
  unlabeled: {
    text: "Labeled Neutral",
    badgeClass: "text-amber-400",
    pillActiveClass: "border-amber-400 text-amber-300 bg-slate-800"
  },
  unreviewed: {
    text: "Propagated & Unwrapped",
    badgeClass: "text-sky-400",
    pillActiveClass: "border-sky-400 text-sky-300 bg-slate-800"
  },
  unconfirmed: {
    text: "Wrapped & Unconfirmed",
    badgeClass: "text-orange-400",
    pillActiveClass: "border-orange-400 text-orange-300 bg-slate-800"
  },
  confirmed: {
    text: "Wrapped & Confirmed",
    badgeClass: "text-emerald-400",
    pillActiveClass: "border-emerald-400 text-emerald-300 bg-slate-800"
  }
};

// FaceScape expression ids are "<number>_<name>" (e.g. "1_neutral",
// "10_dimpler") -- a plain string sort puts "10_dimpler" before "2_smile"
// (lexicographic), so this pulls out the leading number to sort on
// numerically instead, putting 1_neutral first as the natural "start here"
// entry. Anything without a leading number (shouldn't happen for this
// dataset) sorts last rather than crashing.
const facescapeExpressionOrder = (expressionId: string): number => {
  const match = expressionId.match(/^(\d+)/);
  return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
};

export function SequenceExplorer() {
  const {
    activeDatasetType,
    setActiveDatasetType,
    dataset,
    isDatasetLoading,
    explorerError,
    selectedActor,
    selectedSequence,
    selectedCameras,
    fetchDataset,
    selectSequence,
    toggleCamera,
    triggerPreprocess,
    goToNextUnlabeledSequence,
    confirmLabeled,
    confirm,
    unconfirmLabeled,
    unconfirm,
    unconfirmConfirmed,
    pendingLabelActions,
    refreshActorStatus,
    refreshingActors,
    updateAllActorStatuses,
    isUpdatingAll
  } = useAnnotationStore((state) => ({
    activeDatasetType: state.activeDatasetType,
    setActiveDatasetType: state.setActiveDatasetType,
    dataset: state.dataset,
    isDatasetLoading: state.isDatasetLoading,
    explorerError: state.explorerError,
    selectedActor: state.selectedActor,
    selectedSequence: state.selectedSequence,
    selectedCameras: state.selectedCameras,
    fetchDataset: state.fetchDataset,
    selectSequence: state.selectSequence,
    toggleCamera: state.toggleCamera,
    triggerPreprocess: state.triggerPreprocess,
    goToNextUnlabeledSequence: state.goToNextUnlabeledSequence,
    confirmLabeled: state.confirmLabeled,
    confirm: state.confirm,
    unconfirmLabeled: state.unconfirmLabeled,
    unconfirm: state.unconfirm,
    unconfirmConfirmed: state.unconfirmConfirmed,
    pendingLabelActions: state.pendingLabelActions,
    refreshActorStatus: state.refreshActorStatus,
    refreshingActors: state.refreshingActors,
    updateAllActorStatuses: state.updateAllActorStatuses,
    isUpdatingAll: state.isUpdatingAll
  }));

  const [actorSearch, setActorSearch] = useState("");
  const [expandedActors, setExpandedActors] = useState<Set<string>>(new Set());
  const [statusFilter, setStatusFilter] = useState<Set<LabelStatus>>(new Set(LABEL_STATUSES));
  const [minActor, setMinActor] = useState("");
  const [maxActor, setMaxActor] = useState("");

  useEffect(() => {
    if (!dataset && !isDatasetLoading) {
      void fetchDataset();
    }
  }, [dataset, isDatasetLoading, fetchDataset]);

  // Keep the selected actor's sequences visible, e.g. after "Jump to next unlabeled".
  useEffect(() => {
    if (!selectedActor) return;
    setExpandedActors((prev) => {
      if (prev.has(selectedActor)) return prev;
      const next = new Set(prev);
      next.add(selectedActor);
      return next;
    });
  }, [selectedActor]);

  const toggleActor = (actorId: string) => {
    setExpandedActors((prev) => {
      const next = new Set(prev);
      if (next.has(actorId)) {
        next.delete(actorId);
      } else {
        next.add(actorId);
      }
      return next;
    });
  };

  const collapseAll = () => setExpandedActors(new Set());

  const toggleStatusFilter = (status: LabelStatus) => {
    setStatusFilter((prev) => {
      const next = new Set(prev);
      if (next.has(status)) {
        next.delete(status);
      } else {
        next.add(status);
      }
      return next;
    });
  };

  // Filters both which actors show up (search text, actor-number range, and
  // whether any of their sequences match the status filter) and which sequences
  // show up within each remaining actor (the status filter alone).
  const filteredActors = useMemo(() => {
    const query = actorSearch.trim().toLowerCase();
    const min = minActor.trim() ? Number.parseInt(minActor, 10) : null;
    const max = maxActor.trim() ? Number.parseInt(maxActor, 10) : null;

    return (dataset?.actors ?? [])
      .filter((actor) => {
        if (query && !actor.actor_id.toLowerCase().includes(query)) return false;
        const actorNum = Number.parseInt(actor.actor_id, 10);
        if (Number.isNaN(actorNum)) return true;
        if (min !== null && !Number.isNaN(min) && actorNum < min) return false;
        if (max !== null && !Number.isNaN(max) && actorNum > max) return false;
        return true;
      })
      .map((actor) => ({
        ...actor,
        sequences: actor.sequences.filter((sequence) => statusFilter.has(sequence.label_status))
      }))
      .filter((actor) => actor.sequences.length > 0);
  }, [dataset, actorSearch, minActor, maxActor, statusFilter]);

  return (
    <aside className="w-80 border-r border-slate-800 bg-slate-900 text-sm flex flex-col">
      <div className="p-4 border-b border-slate-800">
        <h2 className="text-lg font-semibold text-slate-100">Sequences</h2>
        {isDatasetLoading && <p className="text-xs text-slate-400">Loading dataset…</p>}
        {explorerError && <p className="text-xs text-rose-400">{explorerError}</p>}
        <div className="mt-2 flex items-center justify-between gap-2">
          <button
            className="text-xs text-brand-300 hover:text-brand-100"
            onClick={goToNextUnlabeledSequence}
          >
            Jump to next unlabeled (Shift+U)
          </button>
          <button
            className="text-xs text-brand-300 hover:text-brand-100 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isUpdatingAll}
            title={`Refresh every ${activeDatasetType} actor's label statuses (bypasses the dataset cache) -- same as pressing "Update" on all of them, one at a time.`}
            onClick={() =>
              void (activeDatasetType === "Ava256" ? useAva256Store.getState().fetchDataset() : updateAllActorStatuses())
            }
          >
            {isUpdatingAll ? "Updating all…" : `Update all (${activeDatasetType})`}
          </button>
        </div>
        <input
          type="text"
          value={actorSearch}
          onChange={(event) => setActorSearch(event.target.value)}
          placeholder="Search actor…"
          className="mt-3 w-full rounded bg-slate-800 border border-slate-700 px-3 py-1.5 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-brand-400"
        />
        <div className="mt-3 flex flex-wrap gap-1">
          {LABEL_STATUSES.map((status) => {
            const meta = LABEL_STATUS_META[status];
            const active = statusFilter.has(status);
            return (
              <button
                key={status}
                onClick={() => toggleStatusFilter(status)}
                className={classNames(
                  "rounded-full border px-2 py-0.5 text-[11px] transition-colors",
                  active ? meta.pillActiveClass : "border-slate-700 text-slate-600 hover:text-slate-400"
                )}
              >
                {meta.text}
              </button>
            );
          })}
        </div>
        <div className="mt-2 flex items-center gap-3">
          <label className="flex items-center gap-1 text-xs text-slate-400">
            Min
            <input
              type="number"
              value={minActor}
              onChange={(event) => setMinActor(event.target.value)}
              placeholder="—"
              className="w-16 rounded bg-slate-800 border border-slate-700 px-1.5 py-0.5 text-slate-100 focus:outline-none focus:border-brand-400"
            />
          </label>
          <label className="flex items-center gap-1 text-xs text-slate-400">
            Max
            <input
              type="number"
              value={maxActor}
              onChange={(event) => setMaxActor(event.target.value)}
              placeholder="—"
              className="w-16 rounded bg-slate-800 border border-slate-700 px-1.5 py-0.5 text-slate-100 focus:outline-none focus:border-brand-400"
            />
          </label>
          <button className="ml-auto text-xs text-brand-300 hover:text-brand-100" onClick={collapseAll}>
            Collapse all
          </button>
        </div>
        <div className="mt-3 flex gap-1 border-t border-slate-800 pt-3">
          {(["Nersemble", "FaceScape", "Ava256"] as const).map((datasetType) => (
            <button
              key={datasetType}
              onClick={() => setActiveDatasetType(datasetType)}
              className={classNames(
                "flex-1 rounded px-2 py-1 text-xs font-medium transition-colors",
                activeDatasetType === datasetType
                  ? "bg-brand-500 text-white"
                  : "bg-slate-800 text-slate-400 hover:text-slate-200"
              )}
            >
              {datasetType === "Ava256" ? "Ava-256" : datasetType}
            </button>
          ))}
        </div>
      </div>
      {activeDatasetType === "Ava256" ? (
        <Ava256ExplorerList statusFilter={statusFilter} search={actorSearch} />
      ) : activeDatasetType === "FaceScape" ? (
        <FaceScapeExplorerList statusFilter={statusFilter} />
      ) : (
      <div className="flex-1 overflow-y-auto">
        {filteredActors.map((actor) => {
          const isExpanded = expandedActors.has(actor.actor_id);
          return (
            <div key={actor.actor_id} className="border-b border-slate-800">
              <div className="w-full flex items-center justify-between px-4 py-2 text-slate-400 uppercase tracking-wide text-xs hover:bg-slate-800/60 hover:text-slate-200">
                <button
                  className="flex-1 flex items-center justify-between text-left"
                  onClick={() => toggleActor(actor.actor_id)}
                >
                  <span>{actor.actor_id}</span>
                  <span className={classNames("transition-transform", isExpanded && "rotate-90")}>▸</span>
                </button>
                <button
                  className="ml-3 normal-case tracking-normal text-brand-300 hover:text-brand-100 disabled:opacity-50 disabled:cursor-not-allowed"
                  disabled={refreshingActors.includes(actor.actor_id)}
                  title="Refresh this actor's label statuses (bypasses the dataset cache) -- use after running wrap_script batch tooling that writes label_tracker.json directly."
                  onClick={(event) => {
                    event.stopPropagation();
                    void refreshActorStatus(actor.actor_id);
                  }}
                >
                  {refreshingActors.includes(actor.actor_id) ? "Updating…" : "Update"}
                </button>
              </div>
              {isExpanded &&
                actor.sequences.map((sequence) => (
                  <div key={sequence.sequence_id} className="px-2 pb-2">
                    <button
                      className={classNames(
                        "w-full text-left rounded px-3 py-2 transition-colors",
                        selectedActor === actor.actor_id && selectedSequence === sequence.sequence_id
                          ? "bg-slate-800 text-white"
                          : "hover:bg-slate-800/60 text-slate-200"
                      )}
                      onClick={() => selectSequence(actor.actor_id, sequence.sequence_id)}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{sequence.sequence_id}</span>
                        <span className={classNames("text-xs", LABEL_STATUS_META[sequence.label_status].badgeClass)}>
                          {LABEL_STATUS_META[sequence.label_status].text}
                        </span>
                      </div>
                      <div className="text-xs text-slate-400">
                        {sequence.cameras.length} cameras · {sequence.has_delegation_video ? "Delegation" : "Needs video"}
                      </div>
                    </button>
                    {sequence.label_status === "unlabeled" && (() => {
                      const pending = pendingLabelActions.includes(labelActionKey(actor.actor_id, sequence.sequence_id));
                      return (
                        <button
                          className="mt-1 w-full rounded px-3 py-1 text-xs bg-amber-900/30 text-amber-300 hover:bg-amber-900/50 disabled:opacity-50 disabled:cursor-not-allowed"
                          disabled={pending}
                          onClick={() => confirmLabeled(actor.actor_id, sequence.sequence_id)}
                        >
                          {pending ? "Confirming…" : "Confirm Labeled"}
                        </button>
                      );
                    })()}
                    {sequence.label_status === "unreviewed" && (() => {
                      const pending = pendingLabelActions.includes(labelActionKey(actor.actor_id, sequence.sequence_id));
                      return (
                        <button
                          className="mt-1 w-full rounded px-3 py-1 text-xs bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                          disabled={pending}
                          onClick={() => unconfirmLabeled(actor.actor_id, sequence.sequence_id)}
                        >
                          {pending ? "Unconfirming…" : "Unconfirm Labeled"}
                        </button>
                      );
                    })()}
                    {/* unreviewed -> unconfirmed is automatic (run_pipeline_batch_all.py),
                        not a button -- see mark_unconfirmed in that script. */}
                    {sequence.label_status === "unconfirmed" && (() => {
                      const pending = pendingLabelActions.includes(labelActionKey(actor.actor_id, sequence.sequence_id));
                      return (
                        <div className="mt-1 flex gap-1">
                          <button
                            className="flex-1 rounded px-3 py-1 text-xs bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled={pending}
                            onClick={() => unconfirm(actor.actor_id, sequence.sequence_id)}
                          >
                            Unconfirm
                          </button>
                          <button
                            className="flex-1 rounded px-3 py-1 text-xs bg-orange-900/30 text-orange-300 hover:bg-orange-900/50 disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled={pending}
                            onClick={() => confirm(actor.actor_id, sequence.sequence_id)}
                          >
                            {pending ? "Confirming…" : "Confirm"}
                          </button>
                        </div>
                      );
                    })()}
                    {sequence.label_status === "confirmed" && (() => {
                      const pending = pendingLabelActions.includes(labelActionKey(actor.actor_id, sequence.sequence_id));
                      return (
                        <button
                          className="mt-1 w-full rounded px-3 py-1 text-xs bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                          disabled={pending}
                          onClick={() => unconfirmConfirmed(actor.actor_id, sequence.sequence_id)}
                        >
                          {pending ? "Unconfirming…" : "Unconfirm"}
                        </button>
                      );
                    })()}
                    {selectedActor === actor.actor_id && selectedSequence === sequence.sequence_id && (
                      <div className="mt-2 space-y-1">
                        {sequence.cameras.map((camera) => (
                          <div key={camera.camera_id} className="flex items-center justify-between px-3 text-xs">
                            <label className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={selectedCameras.includes(camera.camera_id)}
                                onChange={() => toggleCamera(camera.camera_id)}
                              />
                              <span>{camera.camera_id}</span>
                            </label>
                            <button
                              className="text-brand-400 hover:text-brand-200"
                              onClick={() => triggerPreprocess(actor.actor_id, sequence.sequence_id, [camera.camera_id])}
                            >
                              Preprocess
                            </button>
                          </div>
                        ))}
                        <button
                          className="w-full text-xs text-left text-brand-300 hover:text-brand-100 px-3"
                          onClick={() => triggerPreprocess(actor.actor_id, sequence.sequence_id)}
                        >
                          Preprocess all cameras
                        </button>
                      </div>
                    )}
                  </div>
                ))}
            </div>
          );
        })}
        {dataset && filteredActors.length === 0 && (
          <div className="p-4 text-slate-400">No actors match "{actorSearch}".</div>
        )}
        {!dataset && !isDatasetLoading && !explorerError && (
          <div className="p-4 text-slate-400">No dataset found.</div>
        )}
      </div>
      )}
    </aside>
  );
}

function FaceScapeExplorerList({ statusFilter }: { statusFilter: Set<LabelStatus> }) {
  const {
    facescapeDataset,
    isFacescapeDatasetLoading,
    facescapeExplorerError,
    selectedFacescapeActor,
    selectedFacescapeExpression,
    selectedFacescapeCameras,
    fetchFacescapeDataset,
    selectFacescapeExpression,
    toggleFacescapeCamera,
    pendingLabelActions,
    confirmFacescape,
    unconfirmFacescape,
    unconfirmConfirmedFacescape
  } = useAnnotationStore((state) => ({
    facescapeDataset: state.facescapeDataset,
    isFacescapeDatasetLoading: state.isFacescapeDatasetLoading,
    facescapeExplorerError: state.facescapeExplorerError,
    selectedFacescapeActor: state.selectedFacescapeActor,
    selectedFacescapeExpression: state.selectedFacescapeExpression,
    selectedFacescapeCameras: state.selectedFacescapeCameras,
    fetchFacescapeDataset: state.fetchFacescapeDataset,
    selectFacescapeExpression: state.selectFacescapeExpression,
    toggleFacescapeCamera: state.toggleFacescapeCamera,
    pendingLabelActions: state.pendingLabelActions,
    confirmFacescape: state.confirmFacescape,
    unconfirmFacescape: state.unconfirmFacescape,
    unconfirmConfirmedFacescape: state.unconfirmConfirmedFacescape
  }));

  const [expandedActors, setExpandedActors] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!facescapeDataset && !isFacescapeDatasetLoading) {
      void fetchFacescapeDataset();
    }
  }, [facescapeDataset, isFacescapeDatasetLoading, fetchFacescapeDataset]);

  useEffect(() => {
    if (!selectedFacescapeActor) return;
    setExpandedActors((prev) => {
      if (prev.has(selectedFacescapeActor)) return prev;
      const next = new Set(prev);
      next.add(selectedFacescapeActor);
      return next;
    });
  }, [selectedFacescapeActor]);

  const toggleActor = (actorId: string) => {
    setExpandedActors((prev) => {
      const next = new Set(prev);
      if (next.has(actorId)) {
        next.delete(actorId);
      } else {
        next.add(actorId);
      }
      return next;
    });
  };

  // Same status-filter behavior as Nersemble's filteredActors in the parent
  // component -- this list was previously unfiltered entirely, ignoring the
  // shared status-filter buttons at the top of the sidebar.
  const filteredActors = useMemo(() => {
    return (facescapeDataset?.actors ?? [])
      .map((actor) => ({
        ...actor,
        expressions: actor.expressions
          .filter((expression) => statusFilter.has(expression.label_status))
          .sort((a, b) => facescapeExpressionOrder(a.expression_id) - facescapeExpressionOrder(b.expression_id))
      }))
      .filter((actor) => actor.expressions.length > 0);
  }, [facescapeDataset, statusFilter]);

  return (
    <div className="flex-1 overflow-y-auto">
      {isFacescapeDatasetLoading && !facescapeDataset && (
        <p className="p-4 text-xs text-slate-400">Loading FaceScape dataset… (first load can take a while over NFS)</p>
      )}
      {facescapeExplorerError && <p className="p-4 text-xs text-rose-400">{facescapeExplorerError}</p>}
      {filteredActors.map((actor) => {
        const isExpanded = expandedActors.has(actor.actor_id);
        return (
          <div key={actor.actor_id} className="border-b border-slate-800">
            <button
              className="w-full flex items-center justify-between px-4 py-2 text-slate-400 uppercase tracking-wide text-xs hover:bg-slate-800/60 hover:text-slate-200 text-left"
              onClick={() => toggleActor(actor.actor_id)}
            >
              <span>{actor.actor_id}</span>
              <span className={classNames("transition-transform", isExpanded && "rotate-90")}>▸</span>
            </button>
            {isExpanded &&
              actor.expressions.map((expression) => (
                <div key={expression.expression_id} className="px-2 pb-2">
                  <button
                    className={classNames(
                      "w-full text-left rounded px-3 py-2 transition-colors",
                      selectedFacescapeActor === actor.actor_id && selectedFacescapeExpression === expression.expression_id
                        ? "bg-slate-800 text-white"
                        : "hover:bg-slate-800/60 text-slate-200"
                    )}
                    onClick={() => selectFacescapeExpression(actor.actor_id, expression.expression_id)}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">
                        {expression.expression_id}
                        {expression.is_invalid && (
                          <span
                            className="ml-2 rounded border border-amber-500/60 px-1 text-[10px] uppercase tracking-wide text-amber-300"
                            title="No origin-space alignment: raw cameras/images, wrapped in its own raw space"
                          >
                            invalid
                          </span>
                        )}
                      </span>
                      <span className={classNames("text-xs", LABEL_STATUS_META[expression.label_status].badgeClass)}>
                        {LABEL_STATUS_META[expression.label_status].text}
                      </span>
                    </div>
                    <div className="text-xs text-slate-400">
                      {expression.cameras.filter((c) => c.has_image).length} cameras
                    </div>
                  </button>
                  {/* unreviewed -> unconfirmed is automatic (run_pipeline.py's
                      mark_unconfirmed call), not a button -- same as Nersemble. */}
                  {expression.label_status === "unconfirmed" && (() => {
                    const pending = pendingLabelActions.includes(facescapeLabelActionKey(actor.actor_id, expression.expression_id));
                    return (
                      <div className="mt-1 flex gap-1">
                        <button
                          className="flex-1 rounded px-3 py-1 text-xs bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                          disabled={pending}
                          onClick={() => unconfirmFacescape(actor.actor_id, expression.expression_id)}
                        >
                          Unconfirm
                        </button>
                        <button
                          className="flex-1 rounded px-3 py-1 text-xs bg-orange-900/30 text-orange-300 hover:bg-orange-900/50 disabled:opacity-50 disabled:cursor-not-allowed"
                          disabled={pending}
                          onClick={() => confirmFacescape(actor.actor_id, expression.expression_id)}
                        >
                          {pending ? "Confirming…" : "Confirm"}
                        </button>
                      </div>
                    );
                  })()}
                  {expression.label_status === "confirmed" && (() => {
                    const pending = pendingLabelActions.includes(facescapeLabelActionKey(actor.actor_id, expression.expression_id));
                    return (
                      <button
                        className="mt-1 w-full rounded px-3 py-1 text-xs bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        disabled={pending}
                        onClick={() => unconfirmConfirmedFacescape(actor.actor_id, expression.expression_id)}
                      >
                        {pending ? "Unconfirming…" : "Unconfirm"}
                      </button>
                    );
                  })()}
                  {selectedFacescapeActor === actor.actor_id && selectedFacescapeExpression === expression.expression_id && (
                    <div className="mt-2 space-y-1">
                      {expression.cameras.filter((c) => c.has_image).map((camera) => (
                        <div key={camera.camera_id} className="flex items-center justify-between px-3 text-xs">
                          <label className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={selectedFacescapeCameras.includes(camera.camera_id)}
                              onChange={() => toggleFacescapeCamera(camera.camera_id)}
                            />
                            <span>{camera.camera_id}</span>
                          </label>
                          {!camera.has_landmarks && <span className="text-slate-500">no landmarks</span>}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
          </div>
        );
      })}
      {facescapeDataset && facescapeDataset.actors.length === 0 && (
        <div className="p-4 text-slate-400">No FaceScape actors found.</div>
      )}
      {facescapeDataset && facescapeDataset.actors.length > 0 && filteredActors.length === 0 && (
        <div className="p-4 text-slate-400">No actors match the current status filter.</div>
      )}
    </div>
  );
}

// Ava-256: a flat list of captures (no expressions). Each capture is one
// keyline video whose frames are labeled Valid / Invalid / Rerun in the player.
function Ava256ExplorerList({ statusFilter, search }: { statusFilter: Set<LabelStatus>; search: string }) {
  const { dataset, isLoading, error, selectedCapture, fetchDataset, selectCapture } = useAva256Store((state) => ({
    dataset: state.dataset,
    isLoading: state.isLoading,
    error: state.error,
    selectedCapture: state.selectedCapture,
    fetchDataset: state.fetchDataset,
    selectCapture: state.selectCapture
  }));

  useEffect(() => {
    if (!dataset && !isLoading) {
      void fetchDataset();
    }
  }, [dataset, isLoading, fetchDataset]);

  const captures = useMemo(() => {
    const query = search.trim().toLowerCase();
    return (dataset?.captures ?? []).filter(
      (capture) =>
        statusFilter.has(capture.label_status) && (!query || capture.capture_id.toLowerCase().includes(query))
    );
  }, [dataset, statusFilter, search]);

  return (
    <div className="flex-1 overflow-y-auto">
      {isLoading && !dataset && <p className="p-4 text-xs text-slate-400">Loading Ava-256 captures…</p>}
      {error && <p className="p-4 text-xs text-rose-400">{error}</p>}
      {captures.map((capture) => {
        const labeled = capture.num_frames - capture.counts.unconfirmed;
        const meta = LABEL_STATUS_META[capture.label_status] ?? LABEL_STATUS_META.unlabeled;
        return (
          <div key={capture.capture_id} className="px-2 py-1 border-b border-slate-800">
            <button
              className={classNames(
                "w-full text-left rounded px-3 py-2 transition-colors disabled:opacity-50",
                selectedCapture === capture.capture_id ? "bg-slate-800 text-white" : "hover:bg-slate-800/60 text-slate-200"
              )}
              disabled={!capture.has_video && !capture.is_done}
              title={!capture.has_video && !capture.is_done ? "No line render video yet" : undefined}
              onClick={() => void selectCapture(capture.capture_id)}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium text-xs break-all">{capture.capture_id}</span>
                <span className={classNames("text-xs shrink-0", meta.badgeClass)}>
                  {capture.is_done ? "Done" : meta.text}
                </span>
              </div>
              <div className="mt-0.5 text-[11px] text-slate-400 tabular-nums">
                {capture.num_frames
                  ? `${labeled}/${capture.num_frames} labeled · ` +
                    `${capture.counts.Valid} valid · ${capture.counts.Invalid} invalid · ${capture.counts.Rerun} rerun`
                  : "not rendered yet"}
              </div>
            </button>
          </div>
        );
      })}
      {dataset && captures.length === 0 && (
        <div className="p-4 text-slate-400">No captures match the current filter.</div>
      )}
    </div>
  );
}
