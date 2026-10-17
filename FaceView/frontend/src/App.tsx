import { MultiviewPlayer } from "@/components/MultiviewPlayer";
import { SequenceExplorer } from "@/components/SequenceExplorer";
import { Toast } from "@/components/Toast";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useSequenceSocket } from "@/hooks/useSequenceSocket";
import { useAnnotationStore } from "@/state/useAnnotationStore";

export default function App() {
  const { statusMessage, selectedSequence } = useAnnotationStore((state) => ({
    statusMessage: state.statusMessage,
    selectedSequence: state.selectedSequence
  }));

  useKeyboardShortcuts();
  useSequenceSocket(selectedSequence);

  return (
    <div className="min-h-screen flex bg-slate-950 text-slate-100">
      <SequenceExplorer />
      <main className="flex-1 flex flex-col">
        <div className="flex-1 p-6 space-y-4">
          <MultiviewPlayer />
        </div>
        <StatusBar message={statusMessage} />
      </main>
      <Toast />
    </div>
  );
}

function StatusBar({ message }: { message?: string }) {
  return (
    <div className="h-10 border-t border-slate-800 flex items-center px-6 text-sm text-slate-300">
      {message ?? "Ready"}
    </div>
  );
}
