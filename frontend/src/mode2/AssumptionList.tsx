import styles from "./AssumptionList.module.css";

interface AssumptionListProps {
  assumptions: string[];
}

/**
 * The list of assumptions the system stated out loud -- an assumed use, an
 * assumed signing date. Shared by the analyzed result and the honest stops so the
 * "here is what I assumed" copy reads and looks the same wherever it appears.
 * Renders nothing when there is nothing to assume.
 */
export function AssumptionList({ assumptions }: AssumptionListProps) {
  if (assumptions.length === 0) {
    return null;
  }
  return (
    <ul className={styles.list}>
      {assumptions.map((assumption, index) => (
        <li key={index} className={styles.item}>
          {assumption}
        </li>
      ))}
    </ul>
  );
}
